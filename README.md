# MLB Hit Predictor — BeatTheStreak

**Patent Pending** | Provisional Patent Filed February 13, 2026 (USPTO)

An XGBoost-based machine learning system that predicts whether MLB batters will record a hit in a given game. Built for MLB's Beat the Streak contest, prediction markets, and commercial licensing.

### Validation Results

**In-Era (2021-2023 train → 2024 backtest):**
- **78.9% top-1 accuracy** (71/90 game dates)
- **21-game prediction streak**

**Cross-Era (2015-2018 train → 2023 full-season backtest):**
- **78.1% top-1 accuracy** (139/178 game dates, interval=1)
- **21-game prediction streak** (identical to in-era)
- 201,157 training rows, 179 dates tested
- Zero overlapping rosters between training and test data

The 0.8% accuracy delta across a 5-year era gap with zero roster overlap demonstrates that the AABBH methodology captures a fundamental, era-independent pattern in hitting behavior.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Infrastructure Deployment](#2-infrastructure-deployment)
3. [System Architecture](#3-system-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Output & Usage](#5-model-output--usage)
6. [API Endpoints](#6-api-endpoints)
7. [Customer-Facing API](#7-customer-facing-api)
8. [Analytics & Dashboards](#8-analytics--dashboards)
9. [Monitoring & Operations](#9-monitoring--operations)
10. [Roadmap](#10-roadmap)
11. [Live Season Results](#11-live-season-results)

---

## 1. Prerequisites

### CI/CD Role

The `mlb-cicd-role` IAM role **must be deployed manually** before any CloudFormation resources can be deployed via GitHub Actions. This role grants the GitHub Actions workflow permission to manage AWS resources.

> **v2.1 note:** If you are upgrading from v2.0, you must add Glue, Athena, SSM, and `lambda:GetLayerVersion` actions to your existing `mlb-cicd-role` before pushing v2.1. Without these, the CFN deploy will silently no-op the analytics resources (the workflow's error-swallower catches the AccessDenied and reports green). Merge the new actions listed below into your existing policy document in the IAM console.

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: IAM Role for GitHub Actions CI/CD Deployments

Resources:
  CICDRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: mlb-cicd-role
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Federated: arn:aws:iam::{YourAccountNumberGoesHere}:oidc-provider/token.actions.githubusercontent.com
            Action: sts:AssumeRoleWithWebIdentity
            Condition:
              StringEquals:
                token.actions.githubusercontent.com:aud: sts.amazonaws.com
              StringLike:
                token.actions.githubusercontent.com:sub:
                  - repo:{YourGitHubUsernameGoesHere}/beatthestreak:ref:refs/heads/*
                  - repo:{YourGitHubUsernameGoesHere}/golfpredictor:ref:refs/heads/*
                  - repo:{YourGitHubUsernameGoesHere}/nbapredictor:ref:refs/heads/*
                  - repo:{YourGitHubUsernameGoesHere}/nflpredictor:ref:refs/heads/*
      Policies:
        - PolicyName: CICDPipelinePolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Sid: GeneralPermissions
                Effect: Allow
                Action:
                  - cloudformation:*
                  - iam:GetRole
                  - iam:PassRole
                  - iam:CreateRole
                  - iam:DeleteRole
                  - iam:PutRolePolicy
                  - iam:DeleteRolePolicy
                  - iam:GetRolePolicy
                  - iam:AttachRolePolicy
                  - iam:DetachRolePolicy
                  - iam:TagRole
                  - iam:UntagRole
                  - iam:ListRolePolicies
                  - iam:ListAttachedRolePolicies
                  - iam:CreateServiceLinkedRole
                  - s3:*
                  - sns:*
                  - kms:GenerateDataKey
                  - kms:Decrypt
                  - ecr:*
                  - ecs:*
                  - ec2:*
                  - elasticloadbalancing:*
                  - lambda:*
                  - lambda:GetLayerVersion
                  - events:*
                  - logs:*
                  # v2.1: Glue + Athena + SSM for analytics pipeline
                  - glue:CreateDatabase
                  - glue:UpdateDatabase
                  - glue:DeleteDatabase
                  - glue:GetDatabase
                  - glue:CreateTable
                  - glue:UpdateTable
                  - glue:DeleteTable
                  - glue:GetTable
                  - athena:CreateWorkGroup
                  - athena:UpdateWorkGroup
                  - athena:DeleteWorkGroup
                  - athena:GetWorkGroup
                  - ssm:GetParameters
                  - ssm:GetParameter
                Resource: "*"
              - Sid: APIGatewayAllOperations
                Effect: Allow
                Action: apigateway:*
                Resource: "*"
                # apigateway:* wildcard required due to known AWS bug where
                # CloudFormation reports TagResource permission errors for
                # API Gateway v2 resources. Tighten when AWS resolves it.
```

I collapsed the verbose per-action lists into wildcards (`cloudformation:*`, `s3:*`, etc.) for readability — your existing policy with the detailed action lists also works fine. Just make sure the Glue, Athena, SSM, and `lambda:GetLayerVersion` actions are added.

### Required Tools

- AWS CLI v2 (configured with appropriate credentials)
- Docker (for local builds and ECR push)
- Node.js / npm (for GitHub Actions)
- Python 3.11+ (for local development)

---

## 2. Infrastructure Deployment

### CI/CD Pipeline Flow

```
VS Code → git push → GitHub Actions → ECR (Docker image) → ECS Fargate (running container)
```

On every push to `main`:

1. GitHub Actions authenticates to AWS via `mlb-cicd-role` (OIDC federation)
2. Builds the Docker image from the repository `Dockerfile`
3. Pushes the image to Amazon ECR
4. Deploys/updates the CloudFormation stack
5. Packages and uploads Lambda functions to S3 (as of v2.1, `check_actuals.zip` bundles `predictions_log.py`)
6. Updates the ECS Fargate service to pull the new image
7. ECS performs a rolling deployment (health check validated before traffic cutover)

### CloudFormation Stack — Single-Stack Architecture

| Resource | Purpose | Gate |
|----------|---------|------|
| **VPC / Subnets / Security Group** | Network config with public IP | Always |
| **ECS Cluster + Service** | Fargate cluster running the API container | Always |
| **ECS Task Definition** | Container config: image, CPU/memory, env vars, logging | Always |
| **S3 Bucket** | Model artifacts, predictions, predictions_log, backtest results | Always |
| **IAM Roles** | Task execution, task role, Lambda roles | Always |
| **CloudWatch Log Groups** | Container + Lambda logs | Always |
| **Glue Database + Table** | `beatthestreak_analytics_{env}.predictions_log` — Athena metadata over parquet | v2.1+ |
| **Athena Workgroup** | `beatthestreak-analytics-{env}` — QuickSight query endpoint | v2.1+ |
| **AWS SDK for pandas Lambda Layer** | Resolved from SSM at deploy time, attached to actuals checker | v2.1+ |
| **EventBridge Rules** | Morning/evening prediction triggers + 10 AM actuals checker | `EnableScheduledPredictions` |
| **Prediction Lambda** | Discovers task IP, triggers `/predict/daily` | `EnableScheduledPredictions` |
| **Actuals Checker Lambda** | Fills previous day hit/miss results + enriches predictions_log | `EnableScheduledPredictions` |
| **SNS Topic + Subscription** | Email alerts for retrain results and accuracy warnings | `AlertEmail` |
| **Retrain Lambda** | Annual pre-season model retraining trigger | `EnableAutoRetrain` |
| **Annual EventBridge Rule** | February 15 trigger for model retraining | `EnableAutoRetrain` |
| **Application Load Balancer** | Internal ALB fronting ECS for API Gateway | `EnablePublicAPI` |
| **API Gateway v2 + VPC Link** | Multi-sport customer-facing API | `EnablePublicAPI` |
| **Custom Domain** | api.beatthestreak.com (optional) | `DomainName` + `CertificateArn` |

### Feature Gates

| Feature | Parameter | Default | When to Enable |
|---------|-----------|---------|----------------|
| Daily Predictions | `EnableScheduledPredictions=true` | false | Production, MLB season |
| Auto-Retrain | `EnableAutoRetrain=true` | false | Production, annual |
| Pipeline Alerts | `AlertEmail=you@email.com` | "" | When email provided |
| Customer API | `EnablePublicAPI=true` | false | Post-licensing |
| Custom Domain | `DomainName` + `CertificateArn` | "" | When domain is ready |
| Pandas Layer Version | `PandasLayerSsmPath` | `/aws/service/aws-sdk-pandas/3.15.1/py3.12/x86_64/layer-arn` | Bump when AWS publishes new release |

All gates require `Environment=production` (except `HasAlertEmail` and `PandasLayerSsmPath`). The test environment never creates gated resources.

### Environment Strategy

- **Test** (`test-cluster`): Development and validation. All gates default to false.
- **Production** (`production-cluster`): Live predictions. Enable gates as needed.

### First-Time Deployment Bootstrap

On first deployment the S3 bucket must exist and Lambda zips must be uploaded **before** pushing to main, because CloudFormation needs them to create the Lambda functions.

```bash
aws s3 mb s3://mlb-hit-predictor-production-bucket --region us-east-1

cd ~/beatthestreak/mlb_predictor/lambda
zip prediction_trigger.zip prediction_trigger.py
zip model_retrainer.zip model_retrainer.py

# v2.1: check_actuals.zip bundles predictions_log.py for analytics enrichment
mkdir -p _build/mlb_predictor
cp check_actuals.py _build/
cp ../predictions_log.py _build/mlb_predictor/
touch _build/mlb_predictor/__init__.py
(cd _build && zip -r ../check_actuals.zip .)
rm -rf _build

aws s3 cp prediction_trigger.zip s3://mlb-hit-predictor-production-bucket/lambda/
aws s3 cp model_retrainer.zip s3://mlb-hit-predictor-production-bucket/lambda/
aws s3 cp check_actuals.zip s3://mlb-hit-predictor-production-bucket/lambda/

git commit --allow-empty -m "chore: trigger deployment" && git push origin main
```

On subsequent deployments the CI/CD pipeline handles Lambda packaging automatically.

### Post-Deployment: Getting the Public IP

```bash
CLUSTER=production-cluster
TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER --query 'taskArns[0]' --output text)
PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text | xargs -I {} aws ec2 describe-network-interfaces \
  --network-interface-ids {} \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
echo $PUBLIC_IP
curl -s http://$PUBLIC_IP:8080/health
```

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEVELOPER WORKFLOW                           │
│  VS Code ──► GitHub (main) ──► GitHub Actions ──► ECR ──► ECS       │
│                                   (mlb-cicd-role)                   │
└─────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ECS FARGATE CLUSTER                          │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                   FastAPI Application                     │      │
│  │  /train/full ─── historical_data.py ─── MLB Stats API     │      │
│  │       ▼                                                   │      │
│  │  feature_engineering.py ──► 26 features + AABBH + ERA     │      │
│  │       ▼                                                   │      │
│  │  model.py ──► XGBoost ──► Calibrated Probabilities        │      │
│  │       ▼                                                   │      │
│  │  /predict/daily ──► daily_predictions.py                  │      │
│  │                            │                              │      │
│  │                            ▼                              │      │
│  │                   predictions_log.py (v2.1)               │      │
│  │  /backtest, /train/retrain-auto, /shap, etc.              │      │
│  └───────────────────────────────────────────────────────────┘      │
└──────────────────────────│──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS SERVICES                                │
│  EventBridge ──► Lambda (prediction_trigger, check_actuals,         │
│                          model_retrainer)                           │
│  S3 ──► predictions_log/ parquet ──► Glue/Athena ──► QuickSight     │
│  SNS ──► Email Alerts                                               │
│  API Gateway v2 ──► VPC Link ──► ALB ──► ECS (gated)                │
│  CloudWatch Logs ──► /ecs/production, /aws/lambda/*                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Map

```
main.py (FastAPI entrypoint, v2.1)
├── historical_data.py      ← MLB Stats API data pulls, schedule enrichment
├── feature_engineering.py  ← 26-feature pipeline, rolling stats, AABBH, ERA tiers
├── model.py                ← XGBoost training, calibration, save/load
├── daily_predictions.py    ← Live prediction pipeline (lineups → features → rank)
├── predictions_log.py      ← v2.1: analytics-grade parquet log for QuickSight
├── backtester.py           ← Historical streak simulation, S3 result storage
├── hyperparameter_tuning.py← RandomizedSearchCV optimization
├── shap_analysis.py        ← SHAP explainability, miss analysis
├── api_routes.py           ← Customer-facing API endpoints (/v1 prefix, gated)
└── lambda/
    ├── prediction_trigger.py ← Daily prediction trigger (EventBridge → Lambda)
    ├── model_retrainer.py    ← Annual retrain trigger (EventBridge → Lambda)
    └── check_actuals.py      ← Morning actuals checker (enriches predictions_log)
```

---

## 4. Feature Engineering

The model uses **26 features** across five categories. All rolling/cumulative features use `shift(1)` to prevent data leakage.

### Batter Features (12)

| Feature | Description |
|---------|-------------|
| `batting_avg_season` | Season-to-date batting average |
| `avg_last_7` | Rolling 7-game hit average |
| `avg_last_14` | Rolling 14-game hit average |
| `avg_last_30` | Rolling 30-game hit average |
| `k_rate` | Season-to-date strikeout rate |
| `bb_rate` | Season-to-date walk rate |
| `multi_hit_freq` | Fraction of games with 2+ hits |
| `avg_abs_per_game` | Average at-bats per game |
| `bat_side_encoded` | Bat side: R=0, L=1, S=2 |
| `games_between_hits` | Consistency metric |
| `hit_streak` | Current consecutive games with a hit |
| `game_num_in_season` | How deep into the season |

### AABBH Features (4) — Patent Pending

| Feature | Description |
|---------|-------------|
| `aabbh` | Total ABs / Total Hits (natural hit frequency) |
| `current_hitless_abs` | Consecutive ABs since last hit |
| `aabbh_ratio` | `current_hitless_abs / aabbh` — the "overdue" signal |
| `in_optimal_zone` | 1 if ratio 0.5–1.0 (81.8% hit rate in backtest) |

### ERA Tier Features (3) — v1.2

| Feature | Description |
|---------|-------------|
| `batter_hit_rate_vs_era_tier` | Hit rate against today's pitcher's ERA tier |
| `batter_ab_vs_era_tier` | Sample size for reliability weighting |
| `era_tier_delta` | Tier-specific rate minus overall rate |

**ERA Tiers:** Elite (<3.00), Good (3.00–3.75), Average (3.75–4.50), Below Average (4.50+)

### Matchup Features (2)

| Feature | Description |
|---------|-------------|
| `is_home` | Home field advantage (1/0) |
| `platoon_advantage` | Handedness advantage vs pitcher |

### Pitcher Features (4)

| Feature | Description |
|---------|-------------|
| `pitcher_era` | Opposing starter's ERA |
| `pitcher_whip` | Opposing starter's WHIP |
| `pitcher_k_per_9` | Opposing starter's K/9 |
| `pitcher_hand_encoded` | Pitcher handedness |

### Environment Features (1)

| Feature | Description |
|---------|-------------|
| `park_factor` | Venue run factor (>1.0 = hitter-friendly) |

> **Phase 2 (planned):** Pitcher-side AABBH cycle features — `pitcher_aabbh_career`, `pitcher_aabbh_recent`, `pitcher_current_hitless_abs`, `pitcher_aabbh_ratio`, `pitcher_in_optimal_zone`, plus a `batter_aabbh_ratio × pitcher_aabbh_ratio` interaction term and `expected_pa_against_starter` derived from lineup slot × starter avg IP. Target: 32 total features, +1.5pp top-1 hit rate gate before promoting from shadow to production. See §10 Roadmap.

---

## 5. Model Output & Usage

### Daily Predictions

The `/predict/daily` endpoint produces a ranked list of every confirmed starter, scored by probability of getting at least one hit.

### S3 Output Structure

```
s3://mlb-hit-predictor-{env}-bucket/
├── predictions/{date}/ and predictions/latest/    ← Legacy CSV + JSON (README pipeline)
├── predictions_log/dt={date}/predictions.parquet  ← v2.1 analytics log (Athena/QuickSight)
├── athena_results/                                ← Athena query output
├── backtest/{season}/{timestamp}/
├── tuning/{timestamp}/
├── analysis/shap/{timestamp}/
├── models/hit_predictor.pkl
└── lambda/ (deployment packages)
```

---

## 6. API Endpoints

### Internal Endpoints

| Method | Endpoint | Description | Runtime |
|--------|----------|-------------|---------|
| `GET` | `/health` | Health check | Instant |
| `POST` | `/train/full` | Train on multiple seasons | 30–60 min |
| `POST` | `/train/tuned` | Hyperparameter tuning | 2–4 hours |
| `POST` | `/train/retrain-auto` | Automated retrain (Lambda-triggered) | 2–3 hours |
| `GET` | `/predict/daily` | Today's ranked predictions | 5–10 min |
| `POST` | `/backtest` | Streak simulation | 3–5 hours |
| `POST` | `/backtest-async` | Background backtest | 3–5 hours |
| `POST` | `/shap` | Feature importance | 5–15 min |

---

## 7. Customer-Facing API

When `EnablePublicAPI=true`, a REST API is exposed via API Gateway. Customers receive an API key and access predictions through the `/v1` prefix.

### Architecture

```
Customer → API Gateway v2 (HTTP API) → VPC Link → ALB → ECS → /v1/*
```

HTTP API Gateway (v2) connects directly to the ALB via VPC Link — no NLB required. Lower latency, ~70% cheaper than REST API, native JWT auth for Stripe paywall integration. Single gateway serves all sports models under one unified API.

### Customer Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/baseball/picks/daily` | Today's MLB hit predictions |
| `GET` | `/v1/baseball/picks/{date}` | Historical MLB predictions |
| `GET` | `/v1/baseball/confidence/{player_id}` | Single player score |
| `GET` | `/v1/golf/picks/tournament` | Current tournament picks |
| `GET` | `/v1/golf/picks/{date}` | Historical golf predictions |
| `GET` | `/v1/basketball/picks/daily` | Today's NBA predictions |
| `GET` | `/v1/basketball/picks/{date}` | Historical NBA predictions |
| `GET` | `/v1/health` | API status |
| `GET` | `/v1/meta` | Model version info |

Auth via JWT (Stripe-integrated). Customer endpoints return predictions only. Training endpoints, model weights, AABBH formula, and internal analytics remain inaccessible.

Full customer documentation: `API_DOCUMENTATION.md`

---

## 8. Analytics & Dashboards

> **New in v2.1.** Complements the legacy `daily_log.json` / `summary.json` pipeline (which still drives the README auto-update in §11) with a flat, BI-friendly prediction log queryable from QuickSight.

### Prediction Log Schema

Every prediction generated by `/predict/daily` is written as one row in a parquet file under `s3://mlb-hit-predictor-{env}-bucket/predictions_log/dt=YYYY-MM-DD/predictions.parquet`. The `check_actuals` Lambda enriches each row with actual game results the following morning.

The table has 37 columns organized into six groups:

- **Identity** — `game_date`, `player_id`, `player_name`, `team`, `opponent`, `venue`, `is_home`
- **Lineup / matchup** — `lineup_position`, `bat_side`, `opp_pitcher`, `opp_pitcher_hand`, `is_doubleheader`, `game_number`, `is_confirmed_starter`
- **Model output** — `rank`, `predicted_prob`, `confidence_pct`, `model_version`
- **Precomputed top-N flags** — `is_top_1`, `is_top_2`, `is_top_5`, `is_top_10`, `is_top_25`, `is_top_50`
- **Key features** — `batting_avg_season`, `avg_last_30`, `aabbh`, `aabbh_ratio`, `in_optimal_zone`, `platoon_advantage`, `pitcher_era`, `pitcher_whip`, `park_factor`
- **Actuals** — `actual_hit`, `actual_ab_count`, `actual_hit_count`, `actuals_resolved_at`

### Athena Partition Projection

The Glue table uses Athena partition projection — new `dt=` partitions are inferred from the date range (`2026-03-25` to `NOW`) the moment they land in S3. No `MSCK REPAIR TABLE`, no Glue crawler, no manual partition management.

### QuickSight Setup

1. New dataset → Athena → workgroup `beatthestreak-analytics-production`
2. Database → `beatthestreak_analytics_production`
3. Table → `predictions_log`
4. Import to SPICE for fast dashboards; schedule hourly refresh

### Validation Query

Run this in Athena after the first day of predictions lands to confirm the pipeline is flowing:

```sql
SELECT dt,
       COUNT(*)                                     AS rows,
       SUM(CAST(actual_hit AS integer))             AS hits,
       ROUND(100.0 * SUM(CAST(actual_hit AS integer)) / COUNT(*), 1) AS hit_rate_pct
FROM beatthestreak_analytics_production.predictions_log
WHERE is_top_1 = true
GROUP BY dt
ORDER BY dt DESC
LIMIT 30;
```

### Model Versioning for Shadow Deploys

Every row is stamped with `model_version` (currently `v1.5`). When new models enter shadow mode (e.g., the Phase 2 pitcher AABBH build will tag as `v2.0_pitcher_aabbh`), they write to the same table alongside production predictions. Dashboard comparisons become a single `GROUP BY model_version` away, with real game outcomes arbitrating which model ships.

### Historical Backfill

To reconstruct the parquet log from the legacy `predictions/YYYY-MM-DD/all_predictions.csv` history:

```bash
python3 scripts/backfill_predictions_log.py
```

Flags: `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--dry-run`, `--no-actuals`, `--force`. Idempotent and safe to re-run.

---

## 9. Monitoring & Operations

### Log Prefixes

| Prefix | Module |
|--------|--------|
| `[PREDICT]` | daily_predictions.py |
| `[TRAIN]` | historical_data.py |
| `[MODEL]` | model.py |
| `[TUNE]` | hyperparameter_tuning.py |
| `[BACKTEST]` | backtester.py |
| `[SHAP]` | shap_analysis.py |
| `[RETRAIN]` | model_retrainer.py / main.py |
| `[ACTUALS]` | check_actuals.py |
| `[predictions_log]` | predictions_log.py |

### Automated Schedules (Production Only)

| Schedule | Lambda / Job | Purpose |
|----------|--------------|---------|
| 10:00 AM ET daily (Mar–Oct) | `actuals_checker` | Fill previous day hit/miss results + enrich predictions_log |
| 11:30 AM ET daily (Mar–Oct) | `prediction_trigger` | Day game lineups |
| 1:00 PM ET daily | GitHub Actions (`update-results.yml`) | Auto-update README live results |
| 5:00 PM ET daily (Mar–Oct) | `prediction_trigger` | Night game lineups |
| Feb 15, 10:00 AM ET (annual) | `model_retrainer` | Pre-season retrain |

### SNS Alerts

When `AlertEmail` is configured:
- **Retrain Started** — Pipeline kicked off
- **Retrain Success** — Model passed ≥ 75% threshold
- **Retrain Warning** — Below threshold, needs review
- **Retrain Failed** — Pipeline error

---

## 10. Roadmap

### Phase 2 — Pitcher AABBH (planned, not yet built)

Extends the AABBH methodology to the pitcher side of the matchup. Currently the model asks "is this batter due for a hit?" but never asks "is this pitcher due to give one up?" Phase 2 closes that gap.

**New features (+6):**
- `pitcher_aabbh_career` — career ABs per hit allowed
- `pitcher_aabbh_recent` — rolling over last 5 starts
- `pitcher_current_hitless_abs` — ABs since last hit allowed
- `pitcher_aabbh_ratio` — the overdue signal, mirror of batter side
- `pitcher_in_optimal_zone` — sweet-spot indicator
- `aabbh_interaction` — `batter_aabbh_ratio × pitcher_aabbh_ratio`

**Derived feature (+1):**
- `expected_pa_against_starter` — from `lineup_position` and pitcher's recent avg IP/start; tells the model "this batter probably gets 2 vs. 3 cracks at the SP"

**Total after Phase 2:** 32 features.

**Ship gate:** Phase 2 ships to production only if a shadow-mode backtest shows **at least +1.5pp top-1 hit rate improvement** over the v2.1 production model on a held-out 2026 window. Shadow predictions write to `predictions_log` under `model_version='v2.0_pitcher_aabbh'` alongside `v1.5` so the comparison is a single `GROUP BY` in QuickSight with real game outcomes arbitrating the decision.

### Phase 3 — DraftKings DFS Optimizer (planned, separate repo)

Multi-sport DK lineup optimizer wrapping `pydfs-lineup-optimizer`. Consumes the `predictions_log` parquet as the MLB projection input, generates GPP-style diversified lineup portfolios with stacking and exposure caps. MLB only for v1; golf second once the abstraction is validated. Backtest framework uses RotoGuru1 historical DK salaries plus four baselines (random valid, chalk, DK-average, our optimizer) to separate projection edge from stacking edge.

---

## Version History

| Version | Date | Changes | OOS Result |
|---------|------|---------|------------|
| v0.1.0 | 2026-01-15 | Initial model with basic batting features | — |
| v0.2.0 | 2026-02-01 | AABBH features, optimal zone discovery | — |
| v0.3.0 | 2026-02-05 | SHAP analysis, async backtest, S3 integration | — |
| v0.4.0 | 2026-02-10 | Hyperparameter tuning endpoint | — |
| v1.0.0 | 2026-02-12 | Baseline: 23 features, default XGBoost params | 78.4% in-sample |
| v1.1.0 | 2026-02-14 | Lineup pre-check, doubleheader handling, AABBH fallback | **76.7%** (69/90), 13-game streak |
| v1.2.0 | 2026-02-15 | ERA tier features (26 features), calibration fix, EventBridge + Lambda | **78.9%** (71/90), **21-game streak** |
| v1.3.0 | 2026-02-16 | Cross-era validation (78.1%, 179 dates), verbose logging fix, auto-retrain Lambda + SNS alerts, API Gateway + ALB + usage plans, customer endpoints, single-stack feature gates | **78.1%** cross-era (139/178), **21-game streak** |
| v2.0.0 | 2026-03-22 | Production deployment — live API feed, EventBridge daily scheduling, actuals checker Lambda, S3→QuickSight pipeline, README auto-update, API Gateway v2 (HTTP API) with multi-sport routes, Top 1/2/5/10 hit rate tracking | **Live as of Opening Day March 25, 2026** |
| v2.1.0 | 2026-04-10 | Analytics-grade prediction log (`predictions_log.py`), Glue database + partition-projected Athena table + Athena workgroup, AWS SDK for pandas Lambda layer attached to actuals checker (512 MB), `ActualsCheckerRole` S3 scope widened to `predictions_log/*`, `backfill_predictions_log.py` historical reconstruction script, deploy workflow bumped to Node 24 actions (checkout@v5, configure-aws-credentials@v5, setup-python@v5), update-live-results workflow fixed with `id-token: write` for OIDC, `model_version` stamping on every prediction to enable shadow A/B deploys for Phase 2, mlb-cicd-role extended with Glue/Athena/SSM/GetLayerVersion permissions | Model unchanged from v2.0 — infrastructure-only release |

---

## 11. Live Season Results

> Results update automatically each morning after the daily prediction pipeline runs.
> Data source: `s3://mlb-hit-predictor-production-bucket/predictions/latest/summary.json`

### 2026 Season — Running Record

| Metric | Value | Updated |
|--------|-------|---------|
| **Top-1 Hit Rate** | — | — |
| **Top-2 Hit Rate** | — | — |
| **Top-5 Hit Rate** | — | — |
| **Top-10 Hit Rate** | — | — |
| **Current Streak (Top-1)** | — | — |
| **Longest Streak (Top-1)** | — | — |
| **Dates Predicted** | — | — |
| **Season Start** | March 25, 2026 | — |

### Daily Log

| Date | Top Pick | Team | Confidence | Top-1 | Top-2 | Top-5 | Top-10 | Streak |
|------|----------|------|------------|-------|-------|-------|--------|--------|
| 2026-04-10 | Wilyer Abreu | Boston Red Sox | 80.0% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-09 | Kevin McGonigle | Detroit Tigers | 75.8% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-08 | Troy Johnston | Colorado Rockies | 82.3% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-07 | Randy Arozarena | Seattle Mariners | 79.6% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-06 | Brice Turang | Milwaukee Brewers | 83.0% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-05 | Oneil Cruz | Pittsburgh Pirates | 79.5% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-04 | Yandy Díaz | Tampa Bay Rays | 81.4% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-03 | Brandon Marsh | Philadelphia Phillies | 79.7% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-02 | Ernie Clement | Toronto Blue Jays | 80.2% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-04-01 | Javier Sanoja | Miami Marlins | 77.1% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-03-31 | Jacob Wilson | Athletics | 71.1% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-03-30 | Jake Burger | Texas Rangers | 73.8% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |
| 2026-03-29 | Logan O'Hoppe | Los Angeles Angels | 80.3% | ⏳ | ⏳ | ⏳ | ⏳ | 0 |

### How Results Are Logged

```
s3://mlb-hit-predictor-production-bucket/
├── predictions/latest/
│   ├── top10_picks.csv       ← Top 10 picks (QuickSight)
│   ├── all_predictions.csv   ← Full ranked list (QuickSight)
│   ├── summary.json          ← Running season metrics (Top 1/2/5/10)
│   └── daily_log.json        ← Per-day pick + result log
├── predictions/{date}/
│   ├── top10_picks.csv
│   ├── all_predictions.csv
│   └── results.json          ← Populated next morning with actuals
└── predictions_log/dt={date}/
    └── predictions.parquet   ← v2.1 analytics log (enriched by check_actuals)
```

### Automated README Updates

A GitHub Action (`update-results.yml`) runs daily at 1 PM ET. It reads `summary.json` and `daily_log.json` from S3 and commits the updated tables above directly to this README. The bot commit uses `[skip ci]` to prevent triggering a redeploy.

To trigger a manual update at any time:

```
GitHub → Actions → Update Live Results → Run workflow
```