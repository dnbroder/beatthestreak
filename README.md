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

> **v2.1 note:** If you are upgrading from v2.0, you must add Glue, Athena, SSM, and `lambda:GetLayerVersion` actions to your existing `mlb-cicd-role`. Without these, the CFN deploy will silently no-op the analytics resources (the workflow's error-swallower catches the AccessDenied and reports green).

> **v2.3 note:** If you are upgrading from v2.1, also add `lambda:PublishVersion` to the policy. The v2.3 deploy workflow calls `update-function-code --publish` on each Lambda after S3 upload to force a code reload (CloudFormation doesn't detect same-key S3 zip content changes, so without this step fresh zips never become live — symptom: Lambda keeps running old code even though a fresh zip exists on S3).

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
                  - lambda:PublishVersion   # v2.3: required for forced Lambda code reload
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
6. **v2.3:** Forces Lambda code reload via `aws lambda update-function-code --publish` against each function, so fresh S3 zips actually get loaded (CFN won't detect same-key content changes on its own)
7. Updates the ECS Fargate service to pull the new image
8. ECS performs a rolling deployment (health check validated before traffic cutover)
9. Smoke test hits `/health`, `/`, and `/test` on the new task

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
| **CloudWatch HEALTH Metric Filter + Alarm** | Counts `[HEALTH] WARNING` lines in actuals checker logs, alerts on 3+ per day | v2.3+ |
| **EventBridge Rules** | Morning/evening prediction triggers + 10 AM actuals checker | `EnableScheduledPredictions` |
| **Prediction Lambda** | Discovers task IP, triggers `/predict/daily` | `EnableScheduledPredictions` |
| **Actuals Checker Lambda** | Fills previous day hit/miss results + enriches predictions_log | `EnableScheduledPredictions` |
| **SNS Topic + Subscription** | Email alerts for retrain results + HEALTH warnings | `AlertEmail` |
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

On subsequent deployments the CI/CD pipeline handles Lambda packaging and code reload automatically.

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
│  │                   predictions_log.py                      │      │
│  │  /backtest, /train/retrain-auto, /shap, etc.              │      │
│  └───────────────────────────────────────────────────────────┘      │
└──────────────────────────│──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS SERVICES                                │
│  EventBridge ──► Lambda (prediction_trigger, check_actuals,         │
│                          model_retrainer)                           │
│  S3 ──► predictions_log/ parquet ──► Glue/Athena ──► QuickSight     │
│  SNS ──► Email Alerts (retrain results + HEALTH warnings)           │
│  CloudWatch ──► Metric Filter ──► Alarm ──► SNS (v2.3+)             │
│  API Gateway v2 ──► VPC Link ──► ALB ──► ECS (gated)                │
│  CloudWatch Logs ──► /ecs/production, /aws/lambda/*                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Map

```
main.py (FastAPI entrypoint, v2.3)
├── historical_data.py      ← MLB Stats API data pulls, schedule enrichment
├── feature_engineering.py  ← 26-feature pipeline, rolling stats, AABBH, ERA tiers
├── model.py                ← XGBoost training, calibration, save/load
├── daily_predictions.py    ← Live prediction pipeline (lineups → features → rank)
├── predictions_log.py      ← Analytics-grade parquet log for QuickSight
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

> **Phase 2 (scrapped April 2026):** Pitcher-side AABBH features were built, trained, and backtested but failed the ship gate. V2 underperformed V1 on top-1 by 4.32pp on 185 held-out 2024 dates. See §10 Roadmap for the full post-mortem.

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

Every row is stamped with `model_version` (currently `v1.5.4`). When new models enter shadow mode (e.g., the Phase 2 pitcher AABBH build will tag as `v2.0_pitcher_aabbh`), they write to the same table alongside production predictions. Dashboard comparisons become a single `GROUP BY model_version` away, with real game outcomes arbitrating which model ships.

### Historical Backfill

To reconstruct the parquet log from the legacy `predictions/YYYY-MM-DD/all_predictions.csv` history:

```bash
python3 scripts/backfill_predictions_log.py
```

Flags: `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--dry-run`, `--no-actuals`, `--force`. Idempotent and safe to re-run.

### Historical Re-score

To re-run a single historical date through the current prediction pipeline (useful when a pipeline bug fix has landed and a specific date needs clean picks):

```bash
python3 scripts/rescore_date.py --date 2026-04-02
```

Rewrites the date's CSVs, parquet partition, and `daily_log.json` entry, then invokes the actuals checker Lambda to resolve the fresh picks against real results. Note: this rewrites history — the new entry reflects what the current pipeline would have produced given the current MLB Stats API state, not a bit-identical replay.

---

## 9. Monitoring & Operations

### Log Prefixes

Every module writes to CloudWatch using a consistent prefix so you can grep for subsystems independently.

| Prefix | Module | Purpose |
|--------|--------|---------|
| `[PREDICT]` | daily_predictions.py | Prediction pipeline trace |
| `[TRAIN]` | historical_data.py | Training data pulls |
| `[MODEL]` | model.py | Model load/save/calibration |
| `[TUNE]` | hyperparameter_tuning.py | Hyperparameter search |
| `[BACKTEST]` | backtester.py | Historical simulation |
| `[SHAP]` | shap_analysis.py | Feature importance |
| `[RETRAIN]` | model_retrainer.py / main.py | Annual retrain |
| `[ACTUALS]` | check_actuals.py | Morning actuals resolution |
| `[predictions_log]` | predictions_log.py | Parquet write / enrich |
| `[HEALTH]` | any module (v2.3+) | Warning-level signals (see below) |

### HEALTH Logging & CloudWatch Alarm (v2.3+)

Every subsystem emits `[HEALTH]` prefixed log lines as a standardized warning channel, independent of operational chatter. Two tiers:

- **Informational** — `[HEALTH] <metric>=<value>` lines always emit on every run. Examples: `pandas_layer_loaded=True`, `confidence_histogram={...}`, `streak_backfill resolved_days=14 ...`. These are leading indicators that let you verify pipeline state via log inspection without running any test code.

- **Warning** — `[HEALTH] WARNING: <reason>` lines only emit when a guard rail fires. Examples: `max count > 1 but dh_teams empty` (doubleheader detection miss), `high skip rate 22%` (MLB Stats API degradation), `zero resolved actuals` (enrichment failed), `schema_drift unexpected_columns=[...]` (upstream feature added without updating LOG_COLUMNS), `pipeline_duration_sec=960 exceeded 900s threshold`.

A CloudWatch Logs metric filter attached to the actuals checker log group counts `[HEALTH] WARNING` occurrences. A CloudWatch alarm fires when this count reaches **3 in a 24-hour window**, routing to the existing `AlertEmail` SNS topic. Informational `[HEALTH]` lines do not trigger the alarm — only explicit warnings do.

**Grep patterns for operators:**

```bash
# All HEALTH signals (informational + warnings) for the actuals Lambda
aws logs tail /aws/lambda/production-actuals-checker --since 24h --filter-pattern '[HEALTH]'

# Only warnings — these are what the alarm counts
aws logs tail /aws/lambda/production-actuals-checker --since 24h --filter-pattern '"[HEALTH] WARNING"'

# Same for the ECS prediction pipeline
aws logs tail /ecs/production --since 24h --filter-pattern '"[HEALTH] WARNING"'
```

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
- **HEALTH Alarm Fired** (v2.3+) — 3+ `[HEALTH] WARNING` lines in 24-hour window on actuals checker

---

## 10. Roadmap

### Phase 2 — Pitcher AABBH (SCRAPPED April 13, 2026)

Pitcher-side AABBH features were empirically validated, fully built, trained, and backtested. They failed the ship gate and were not promoted.

**What was built:** 11 new pitcher-side features across two families:

- **AABBH family (8):** `pitcher_aabbh_career`, `pitcher_aabbh_recent`, `pitcher_current_hitless_abs`, `pitcher_aabbh_ratio`, `pitcher_in_optimal_zone` (inverted vs batter side — low ratio = vulnerable), `pitcher_aabbh_cv`, `aabbh_interaction` (batter ratio × pitcher ratio), `expected_pa_against_starter`
- **SitCode family (3):** `pitcher_late_inning_degradation` (OPS 7th+ vs 1–6), `pitcher_pitch_count_fade` (OPS pitches 76+ vs first 75), `pitcher_first_inning_disadvantage` (OPS 1st inning vs 1–6)

**Empirical foundation (all confirmed):**
- Zone discovery v1 on 38 pitchers across 3,636 games showed the vulnerable zone is at LOW ratios (0.5–1.0 bucket, 25.94% hit rate vs 24.20% baseline, 3.19pp spread) — inverted from the batter side where high ratios signal overdue hitters.
- Zone discovery v2 de-confounded variance via coefficient of variation (stddev / career_aabbh) and preserved 90% of the signal after controlling for pitcher skill. LOW-CV pitchers were reliably 2.09pp more hittable than HIGH-CV pitchers.
- SitCode features probed on 23 pitchers showed meaningful spread (0.10–0.23 stddev), low cross-feature correlation (max \|r\| = 0.31), and acceptable skill-confound independence.

**Training (201,014 rows, 2021–2024 seasons, held-out 2024):**
- V1 baseline (26 features): accuracy 0.5961, AUC 0.6230
- V2 Phase 2 (37 features): accuracy 0.5953, AUC 0.6231
- Essentially identical aggregate metrics.

**Backtest on 185 held-out 2024 dates (pure holdout, training cutoff 2023-09-24):**

| Metric | V1 Baseline | V2 Phase 2 | Delta |
|--------|-------------|------------|-------|
| Top-1 Hit Rate | **83.24%** | 78.92% | **-4.32pp** |
| Top-2 Hit Rate | 95.14% | 95.68% | +0.54pp |
| Top-5 Hit Rate | 98.92% | 99.46% | +0.54pp |
| Top-10 Hit Rate | 100.00% | 100.00% | +0.00pp |
| Longest Streak (Top-1) | 17 | 12 | -5 |

**Verdict:** FAIL against the +1.5pp top-1 ship gate. Phase 2 was not promoted.

**What was learned:**
- V1 baseline trained on an extended 2021–2024 window reaches **83.24% top-1 on 185 unseen 2024 dates** — meaningfully above the published 78.9% (2021–2023 → 2024) baseline. Extra year of training data is the dominant factor, not new features.
- V2 improved top-2/5/10 modestly (+0.54pp each). The pitcher features are being used by the model and help identify hitters in aggregate — but they push the wrong batter to the #1 slot more often than V1 does.
- The 0.5–1.0 AABBH "vulnerable zone" effect for pitchers is a genuine empirical finding and remains part of the patent claim record. It just didn't translate to top-1 lift in a deployed model.
- Batter-side AABBH appears to be at or near its ceiling for top-1 hit prediction on MLB. The path forward is cross-sport (NFL, golf, NBA) rather than deeper feature stacking on MLB.

**Patent implications:** The provisional filing priority is preserved. The inverted-zone finding on the pitcher side is a substantive empirical result even though we declined to deploy it. If the methodology is later revisited — different model architecture, different target (top-2+ or streak length), different sport — the discovery work remains usable.

### Phase 3 — DraftKings DFS Optimizer (planned, separate repo)

Multi-sport DK lineup optimizer wrapping `pydfs-lineup-optimizer`. Consumes the `predictions_log` parquet as the MLB projection input, generates GPP-style diversified lineup portfolios with stacking and exposure caps. MLB only for v1; golf second once the abstraction is validated. Backtest framework uses RotoGuru1 historical DK salaries plus four baselines (random valid, chalk, DK-average, our optimizer) to separate projection edge from stacking edge.

### AHS — Deferred

"Average Hot Streak" cycles were considered but deferred. Research is mixed on whether streak-shaped features add signal beyond what existing rolling-window features already capture. Decision revisited never — given Phase 2's outcome, the investment case for further MLB feature expansion is weak.

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
| v1.3.0 | 2026-02-16 | Cross-era validation (78.1%, 179 dates), auto-retrain, API Gateway, single-stack feature gates | **78.1%** cross-era, **21-game streak** |
| v2.0.0 | 2026-03-22 | Production deployment — live API feed, EventBridge daily scheduling, actuals checker Lambda, S3→QuickSight pipeline, README auto-update, API Gateway v2, Top 1/2/5/10 tracking | **Live as of Opening Day March 25, 2026** |
| v2.1.0 | 2026-04-10 | Analytics-grade prediction log (`predictions_log.py`), Glue database + Athena table + workgroup, AWS SDK for pandas Lambda layer, backfill script, OIDC workflow fix, `model_version` stamping, Glue/Athena/SSM IAM | Infrastructure-only release |
| v2.2.0 | 2026-04-11 | April 2 output-side dedup fix (`drop_duplicates` in `daily_predictions.py`), diacritic name-strip in `check_actuals.py` (Jesús Sánchez class), `append_actuals` moved above idempotency early-return, `rescore_date.py` one-off rescore script | Model unchanged, data integrity fixes |
| v2.3.0 | 2026-04-12 | HEALTH logging convention across all modules (`[HEALTH]` prefix), CloudWatch metric filter + alarm (3+ warnings/24h), cold-start pandas layer probe, confidence histogram, skip-reason tracking, schema drift guard, pipeline duration guard (>900s WARN), per-day Top-1 streak backfill in `update_summary()` fixing the misleading Daily Log streak column, anchor-based `update_readme_results.py` bot, deploy workflow forces Lambda code reload via `update-function-code --publish` (CFN doesn't detect same-key S3 content changes), `lambda:PublishVersion` added to `mlb-cicd-role` | Model unchanged, observability + deploy reliability release |
| v2.4.0 | 2026-04-13 | Phase 2 pitcher AABBH features built, trained on 201,014 rows across 2021–2024, backtested on 185 held-out 2024 dates. FAILED ship gate: V2 top-1 78.92% vs V1 83.24% (-4.32pp). Reverted cleanly — removed `pitcher_features.py`, `train_phase2.py`, `scripts/backtest_phase2.py`; reverted `daily_predictions.py`, `feature_engineering.py`, `historical_data.py` to pre-Phase-2 state. Extended training window discovered as side effect: V1 baseline on 2021–2024 training reaches 83.24% top-1, up from 78.9% with 2021–2023 training. See §10 Roadmap for full post-mortem. | **V1 83.24% top-1 on 185 held-out 2024 dates (extended window), Phase 2 scrapped** |

---

## 11. Live Season Results

> Results update automatically each morning after the daily prediction pipeline runs.
> Data source: `s3://mlb-hit-predictor-production-bucket/predictions/latest/summary.json`

### 2026 Season — Running Record

<!-- BTS:RUNNING_RECORD_START -->
| Metric | Value | Updated |
|--------|-------|---------|
| **Top-1 Hit Rate** | 70.4% | 2026-08-08 17:26 ET |
| **Top-2 Hit Rate** | 92.8% | 2026-08-08 17:26 ET |
| **Top-5 Hit Rate** | 100.0% | 2026-08-08 17:26 ET |
| **Top-10 Hit Rate** | 100.0% | 2026-08-08 17:26 ET |
| **Current Streak (Top-1)** | 7 | 2026-08-08 17:26 ET |
| **Longest Streak (Top-1)** | 9 | 2026-08-08 17:26 ET |
| **Dates Predicted** | 125 | 2026-08-08 17:26 ET |
| **Season Start** | March 25, 2026 | — |
<!-- BTS:RUNNING_RECORD_END -->

### Daily Log

<!-- BTS:DAILY_LOG_START -->
| Date | Top Pick | Team | Confidence | Top-1 | Top-2 | Top-5 | Top-10 | Top-1 Streak |
|------|----------|------|------------|-------|-------|-------|--------|--------------|
| 2026-08-08 | Luis Arraez | Philadelphia Phillies | 79.1% | ⏳ | ⏳ | ⏳ | ⏳ | — |
| 2026-08-07 | Kyle Stowers | Miami Marlins | 75.8% | ✅ | ✅ | ✅ | ✅ | 7 |
| 2026-08-06 | Luis Arraez | Philadelphia Phillies | 75.4% | ✅ | ✅ | ✅ | ✅ | 6 |
| 2026-08-05 | Luis Arraez | Philadelphia Phillies | 75.4% | ✅ | ✅ | ✅ | ✅ | 5 |
| 2026-08-04 | Trea Turner | Philadelphia Phillies | 75.9% | ✅ | ✅ | ✅ | ✅ | 4 |
| 2026-08-03 | Jonathan Aranda | Tampa Bay Rays | 76.9% | ✅ | ✅ | ✅ | ✅ | 3 |
| 2026-08-02 | Trea Turner | Philadelphia Phillies | 70.3% | ✅ | ✅ | ✅ | ✅ | 2 |
| 2026-08-01 | Drake Baldwin | Atlanta Braves | 75.2% | ✅ | ✅ | ✅ | ✅ | 1 |
| 2026-07-31 | José Ramírez | Cleveland Guardians | 72.8% | ❌ | ❌ | ✅ | ✅ | 0 |
| 2026-07-30 | Matt Olson | Atlanta Braves | 71.6% | ❌ | ✅ | ✅ | ✅ | 0 |
| 2026-07-29 | Luis Arraez | San Francisco Giants | 74.0% | ✅ | ✅ | ✅ | ✅ | 2 |
| 2026-07-28 | Shohei Ohtani | Los Angeles Dodgers | 71.0% | ✅ | ✅ | ✅ | ✅ | 1 |
| 2026-07-27 | CJ Abrams | Washington Nationals | 77.0% | ❌ | ✅ | ✅ | ✅ | 0 |
| 2026-07-26 | Jackson Chourio | Milwaukee Brewers | 73.2% | ✅ | ✅ | ✅ | ✅ | 4 |
| 2026-07-25 | Luis Arraez | San Francisco Giants | 77.5% | ✅ | ✅ | ✅ | ✅ | 3 |
| 2026-07-24 | Luis Arraez | San Francisco Giants | 76.1% | ✅ | ✅ | ✅ | ✅ | 2 |
| 2026-07-23 | Mauricio Dubón | Atlanta Braves | 69.0% | ✅ | ✅ | ✅ | ✅ | 1 |
| 2026-07-22 | Luis Arraez | San Francisco Giants | 73.2% | ❌ | ❌ | ✅ | ✅ | 0 |
| 2026-07-21 | James Wood | Washington Nationals | 75.0% | ✅ | ✅ | ✅ | ✅ | 1 |
| 2026-07-20 | Luis Arraez | San Francisco Giants | 74.2% | ❌ | ✅ | ✅ | ✅ | 0 |
| 2026-07-19 | Elly De La Cruz | Cincinnati Reds | 72.7% | ✅ | ✅ | ✅ | ✅ | 1 |
| 2026-07-18 | Sal Stewart | Cincinnati Reds | 74.8% | ❌ | ✅ | ✅ | ✅ | 0 |
| 2026-07-17 | JJ Wetherholt | St. Louis Cardinals | 70.8% | ✅ | ✅ | ✅ | ✅ | 4 |
| 2026-07-16 | Juan Soto | New York Mets | 67.4% | ✅ | ✅ | ✅ | ✅ | 3 |
| 2026-07-12 | Wyatt Langford | Texas Rangers | 79.8% | ✅ | ✅ | ✅ | ✅ | 2 |
| 2026-07-11 | Brandon Lowe | Pittsburgh Pirates | 72.7% | ✅ | ✅ | ✅ | ✅ | 1 |
| 2026-07-10 | Pete Crow-Armstrong | Chicago Cubs | 80.6% | ❌ | ✅ | ✅ | ✅ | 0 |
| 2026-07-09 | Trea Turner | Philadelphia Phillies | 73.8% | ❌ | ❌ | ✅ | ✅ | 0 |
| 2026-07-08 | Drake Baldwin | Atlanta Braves | 70.9% | ✅ | ✅ | ✅ | ✅ | 1 |
| 2026-07-07 | Jackson Chourio | Milwaukee Brewers | 76.7% | ❌ | ✅ | ✅ | ✅ | 0 |
<!-- BTS:DAILY_LOG_END -->

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

A GitHub Action (`update-results.yml`) runs daily at 1 PM ET. It reads `summary.json` and `daily_log.json` from S3 and commits the updated tables above directly to this README.

As of v2.3 the bot uses HTML comment anchors (`<!-- BTS:RUNNING_RECORD_START -->` / `_END` and `<!-- BTS:DAILY_LOG_START -->` / `_END`) to identify the managed regions. Everything outside those anchors is untouched. The bot commit uses `[skip ci]` to prevent triggering a redeploy.

To trigger a manual update at any time:

```
GitHub → Actions → Update Live Results → Run workflow
```