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
8. [Monitoring & Operations](#8-monitoring--operations)
9. [Live Season Results](#9-live-season-results)

---

## 1. Prerequisites

### CI/CD Role

The `mlb-cicd-role` IAM role **must be deployed manually** before any CloudFormation resources can be deployed via GitHub Actions. This role grants the GitHub Actions workflow permission to manage AWS resources.

**Deploy via CloudFormation Console or CLI before any other stack:**

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
                token.actions.githubusercontent.com:sub: repo:{YourGitHubAccountUsernameGoesHere}/{YourGitHubRepositoryNameForThisApplicationGoesHere}:ref:refs/heads/main
      Policies:
        - PolicyName: CICDPipelinePolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - cloudformation:CreateStack
                  - cloudformation:UpdateStack
                  - cloudformation:DescribeStacks
                  - cloudformation:DescribeStackEvents
                  - cloudformation:GetTemplateSummary
                  - cloudformation:ValidateTemplate
                  - cloudformation:CreateChangeSet
                  - cloudformation:ExecuteChangeSet
                  - cloudformation:DeleteStack
                  - iam:GetRole
                  - iam:PassRole
                  - s3:PutObject
                  - s3:GetObject
                  - s3:ListBucket
                  - kms:GenerateDataKey
                  - kms:Decrypt
                  - ecr:GetAuthorizationToken
                  - ecr:InitiateLayerUpload
                  - ecr:UploadLayerPart
                  - ecr:CompleteLayerUpload
                  - ecr:BatchCheckLayerAvailability
                  - ecr:GetDownloadUrlForLayer
                  - ecr:PutImage
                  - ecr:BatchGetImage
                  - ecs:UpdateService
                  - ecs:ListServices
                  - ecs:DescribeServices
                  - ecs:ListTasks
                  - ecs:DescribeTasks
                  - ec2:DescribeNetworkInterfaces
                Resource: "*"
```

### Required Tools

- AWS CLI v2 (configured with appropriate credentials)
- Docker (for local builds and ECR push)
- Node.js / npm (for GitHub Actions)
- Python 3.11+ (for local development)

---

## 2. Infrastructure Deployment

### CI/CD Pipeline Flow

All deployments follow this path:

```
VS Code → git push → GitHub Actions → ECR (Docker image) → ECS Fargate (running container)
```

On every push to `main`:

1. GitHub Actions authenticates to AWS via `mlb-cicd-role` (OIDC federation)
2. Builds the Docker image from the repository `Dockerfile`
3. Pushes the image to Amazon ECR
4. Packages and uploads Lambda functions to S3
5. Deploys/updates the CloudFormation stack
6. Updates the ECS Fargate service to pull the new image
7. ECS performs a rolling deployment (health check validated before traffic cutover)

### CloudFormation Stack — Single-Stack Architecture

All infrastructure is defined in a single `infrastructure.yaml` template with feature gates. Resources are conditionally created based on parameters, so you only pay for what you enable.

| Resource | Purpose | Gate |
|----------|---------|------|
| **VPC / Subnets / Security Group** | Network config with public IP | Always |
| **ECS Cluster + Service** | Fargate cluster running the API container | Always |
| **ECS Task Definition** | Container config: image, CPU/memory, env vars, logging | Always |
| **S3 Bucket** | Model artifacts, predictions, backtest results | Always |
| **IAM Roles** | Task execution, task role, Lambda roles | Always |
| **CloudWatch Log Groups** | Container + Lambda logs | Always |
| **EventBridge Rules** | Two daily prediction triggers (morning + evening) | `EnableScheduledPredictions` |
| **Prediction Lambda** | Discovers task IP, checks schedule, triggers `/predict/daily` | `EnableScheduledPredictions` |
| **SNS Topic + Subscription** | Email alerts for retrain results and accuracy warnings | `AlertEmail` |
| **Retrain Lambda** | Annual pre-season model retraining trigger | `EnableAutoRetrain` |
| **Annual EventBridge Rule** | February 15 trigger for model retraining | `EnableAutoRetrain` |
| **Application Load Balancer** | Internal ALB fronting ECS for API Gateway | `EnablePublicAPI` |
| **API Gateway + VPC Link** | Customer-facing REST API with usage plans | `EnablePublicAPI` |
| **Usage Plans (3 tiers)** | Basic / Pro / Enterprise rate limiting | `EnablePublicAPI` |
| **Custom Domain** | api.beatthestreak.com (optional) | `DomainName` + `CertificateArn` |

### Feature Gates

| Feature | Parameter | Condition | Default | When to Enable |
|---------|-----------|-----------|---------|----------------|
| Daily Predictions | `EnableScheduledPredictions=true` | `ScheduleEnabled` | false | Production, MLB season |
| Auto-Retrain | `EnableAutoRetrain=true` | `RetrainEnabled` | false | Production, annual |
| Pipeline Alerts | `AlertEmail=you@email.com` | `HasAlertEmail` | "" | When email provided |
| Customer API | `EnablePublicAPI=true` | `APIEnabled` | false | Post-licensing |
| Custom Domain | `DomainName` + `CertificateArn` | `HasCustomDomain` | "" | When domain is ready |

All gates require `Environment=production` (except `HasAlertEmail`). The test environment never creates gated resources.

### Environment Strategy

Two parallel environments share the same architecture:

- **Test** (`test-cluster`): Development and validation. Train, tune, backtest here. All gates default to false.
- **Production** (`prod-cluster`): Live predictions for the season. Enable gates as needed.

### Post-Deployment: Getting the Public IP

```bash
CLUSTER=test-cluster
TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER --query 'taskArns[0]' --output text)
PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text | xargs -I {} aws ec2 describe-network-interfaces \
  --network-interface-ids {} \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)
echo $PUBLIC_IP
curl -s http://$PUBLIC_IP:8080/health
```

> **Note:** The `$PUBLIC_IP` variable is lost when you close your terminal session. Re-run the above commands in any new session.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEVELOPER WORKFLOW                           │
│                                                                     │
│  VS Code ──► GitHub (main) ──► GitHub Actions ──► ECR ──► ECS       │
│                                   (mlb-cicd-role)                   │
└─────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ECS FARGATE CLUSTER                          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                   FastAPI Application                     │      │
│  │                                                           │      │
│  │  /train/full ─── historical_data.py ─── MLB Stats API     │      │
│  │       │              (2000-2024)                          │      │
│  │       ▼                                                   │      │
│  │  feature_engineering.py ──► 26 features + AABBH + ERA     │      │
│  │       │                      tier features                │      │
│  │       ▼                                                   │      │
│  │  model.py ──► XGBoost ──► Calibrated Probabilities        │      │
│  │       │                                                   │      │
│  │       ▼                                                   │      │
│  │  /predict/daily ──► daily_predictions.py                  │      │
│  │  /backtest ──► backtester.py                              │      │
│  │  /train/retrain-auto ──► train → backtest → validate      │      │
│  │  /train/tuned ──► hyperparameter_tuning.py                │      │
│  │  /shap ──► shap_analysis.py                               │      │
│  └───────────────────────────────────────────────────────────┘      │
│                          │                                          │
└──────────────────────────│──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS SERVICES                                │
│                                                                     │
│  EventBridge ──► Lambda (prediction_trigger, model_retrainer)       │
│  S3 ──► QuickSight Dashboard                                       │
│  SNS ──► Email Alerts (retrain results, accuracy warnings)          │
│  API Gateway ──► VPC Link ──► ALB ──► ECS (customer API, gated)     │
│  CloudWatch Logs ──► /ecs/test, /ecs/prod, /aws/lambda/*            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Map

```
main.py (FastAPI entrypoint, v1.3)
├── historical_data.py      ← MLB Stats API data pulls, schedule enrichment
├── feature_engineering.py  ← 26-feature pipeline, rolling stats, AABBH, ERA tiers
├── model.py                ← XGBoost training, calibration, save/load
├── daily_predictions.py    ← Live prediction pipeline (lineups → features → rank)
├── backtester.py           ← Historical streak simulation, S3 result storage
├── hyperparameter_tuning.py← RandomizedSearchCV optimization
├── shap_analysis.py        ← SHAP explainability, miss analysis
├── api_routes.py           ← Customer-facing API endpoints (/v1 prefix, gated)
└── lambda/
    ├── prediction_trigger.py ← Daily prediction trigger (EventBridge → Lambda)
    └── model_retrainer.py    ← Annual retrain trigger (EventBridge → Lambda)
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

---

## 5. Model Output & Usage

### Daily Predictions

The `/predict/daily` endpoint produces a ranked list of every confirmed starter, scored by probability of getting at least one hit.

### S3 Output Structure

```
s3://mlb-hit-predictor-{env}-bucket/
├── predictions/{date}/ and predictions/latest/
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
Customer → API Gateway (auth + rate limit) → VPC Link → ALB → ECS → /v1/*
```

### Usage Plan Tiers

| Tier | Daily Picks | API Calls/Day | Fields |
|------|-------------|---------------|--------|
| **Basic** | Top 5 | 100 | Name, confidence |
| **Pro** | Top 10 | 1,000 | + batting avg, AABBH, pitcher ERA |
| **Enterprise** | Top 50 | 50,000 | All 26 features |

### Customer Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/picks/daily` | Today's ranked predictions |
| `GET` | `/v1/picks/{date}` | Historical predictions |
| `GET` | `/v1/confidence/{player_id}` | Single player score |
| `GET` | `/v1/health` | API status |
| `GET` | `/v1/meta` | Model version info |

Customer endpoints return predictions only. Training endpoints, model weights, AABBH formula, and all internal analytics remain inaccessible.

Full customer documentation: `API_DOCUMENTATION.md`

---

## 8. Monitoring & Operations

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

### Automated Schedules (Production Only)

| Schedule | Lambda | Purpose |
|----------|--------|---------|
| Morning (11:30 AM ET, Mar–Oct) | `prediction_trigger` | Day game lineups |
| Evening (5:00 PM ET, Mar–Oct) | `prediction_trigger` | Night game lineups |
| Annual (Feb 15, 10:00 AM ET) | `model_retrainer` | Pre-season retrain |

### SNS Alerts

When `AlertEmail` is configured:
- **Retrain Started** — Pipeline kicked off
- **Retrain Success** — Model passed ≥ 75% threshold
- **Retrain Warning** — Below threshold, needs review
- **Retrain Failed** — Pipeline error

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
| v2.0.0 | 2026-03-22 | Production deployment — live API feed, EventBridge daily scheduling, S3→QuickSight pipeline, summary.json + daily_log.json for README auto-update, deploy-prod.yml CI/CD workflow, CloudFormation error handling | **Live as of Opening Day March 25, 2026** |

---

## 9. Live Season Results

> Results update automatically each morning after the daily prediction pipeline runs.
> Data source: `s3://mlb-hit-predictor-production-bucket/predictions/latest/summary.json`

### 2026 Season — Running Record

| Metric | Value | Updated |
|--------|-------|---------|
| **Top-1 Hit Rate** | — | — |
| **Current Streak** | — | — |
| **Longest Streak** | — | — |
| **Dates Predicted** | — | — |
| **Season Start** | March 27, 2026 | — |

### Daily Log

| Date | Top Pick | Team | Confidence | Result | Streak |
|------|----------|------|------------|--------|--------|
| *Populates after Opening Day (March 27, 2026)* | | | | | |

### How Results Are Logged

Each morning after `/predict/daily` runs, the pipeline writes to S3:

```
s3://mlb-hit-predictor-production-bucket/
├── predictions/latest/
│   ├── top10_picks.csv       ← Top 10 picks (QuickSight)
│   ├── all_predictions.csv   ← Full ranked list (QuickSight)
│   ├── summary.json          ← Running season metrics
│   └── daily_log.json        ← Per-day pick + result log
└── predictions/{date}/
    ├── top10_picks.csv
    ├── all_predictions.csv
    └── results.json          ← Populated next morning with actuals
```

### Automated README Updates

A GitHub Action (`update-results.yml`) runs daily at 1 PM ET. It reads `summary.json` and `daily_log.json` from S3 and commits the updated tables above directly to this README. The bot commit uses `[skip ci]` to prevent triggering a redeploy.

To trigger a manual update at any time:
```
GitHub → Actions → Update Live Results → Run workflow
```