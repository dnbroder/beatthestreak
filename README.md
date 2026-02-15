# MLB Hit Predictor — BeatTheStreak

**Patent Pending** | Provisional Patent Filed February 13, 2026 (USPTO)

An XGBoost-based machine learning system that predicts whether MLB batters will record a hit in a given game. Built for MLB's Beat the Streak contest and prediction market applications.

**Out-of-sample validation (2024 season, never seen during training):**
- **78.9% top-1 accuracy** (71/90 game dates)
- **21-game prediction streak**
- +13.9 percentage points over MLB baseline (65%)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Infrastructure Deployment](#2-infrastructure-deployment)
3. [System Architecture](#3-system-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Output & Usage](#5-model-output--usage)
6. [API Endpoints](#6-api-endpoints)
7. [Monitoring & Operations](#7-monitoring--operations)

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
4. Updates the ECS Fargate service to pull the new image
5. ECS performs a rolling deployment (health check validated before traffic cutover)

### CloudFormation Stack

The infrastructure is defined in a single CloudFormation template that provisions:

| Resource | Purpose |
|----------|---------|
| **ECR Repository** | Stores Docker images for the predictor service |
| **ECS Cluster** | Fargate cluster running the API container |
| **ECS Task Definition** | Container config: image, CPU/memory, environment variables, logging |
| **ECS Service** | Maintains desired task count, health checks, rolling deploys |
| **S3 Bucket** | Stores model artifacts, backtest results, daily predictions, tuning logs |
| **IAM Roles** | Task execution role (ECR pull, CloudWatch), task role (S3 read/write), Lambda role |
| **CloudWatch Log Group** | Container stdout/stderr and Lambda logs for monitoring |
| **VPC / Security Group** | Network config with public IP for API access |
| **EventBridge Rules** | Two daily triggers for automated predictions (production only) |
| **Lambda Function** | Discovers Fargate task IP, checks MLB schedule, triggers `/predict/daily` |

### Environment Strategy

Two parallel environments share the same architecture:

- **Test** (`test-cluster`): Development and validation. Train, tune, backtest here.
- **Production** (`prod-cluster`): Live predictions for the season. Only promoted code runs here.

The `ENVIRONMENT` environment variable controls S3 bucket naming (`mlb-hit-predictor-test-bucket` vs `mlb-hit-predictor-prod-bucket`) and is set in the ECS task definition.

### Post-Deployment: Getting the Public IP

After every deployment, the ECS Fargate task receives a new public IP. You must retrieve it before hitting any endpoints:

```bash
# Set your cluster name (test-cluster or prod-cluster)
CLUSTER=test-cluster

# Get the task ARN and public IP
TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER --query 'taskArns[0]' --output text)

PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text | xargs -I {} aws ec2 describe-network-interfaces \
  --network-interface-ids {} \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo $PUBLIC_IP

# Verify the container is healthy
curl -s http://$PUBLIC_IP:8080/health
```

> **Note:** The `$PUBLIC_IP` variable is lost when you close your terminal session. Re-run the above commands in any new session.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEVELOPER WORKFLOW                           │
│                                                                     │
│  VS Code ──► GitHub (main) ──► GitHub Actions ──► ECR ──► ECS      │
│                                   (mlb-cicd-role)                   │
└─────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ECS FARGATE CLUSTER                          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │                   FastAPI Application                      │      │
│  │                                                           │      │
│  │  /train/full ─── historical_data.py ─── MLB Stats API     │      │
│  │       │              (2000-2024)                           │      │
│  │       ▼                                                   │      │
│  │  feature_engineering.py ──► 26 features + AABBH + ERA     │      │
│  │       │                      tier features                │      │
│  │       ▼                                                   │      │
│  │  model.py ──► XGBoost ──► Calibrated Probabilities        │      │
│  │       │                                                   │      │
│  │       ▼                                                   │      │
│  │  /predict/daily ──► daily_predictions.py                  │      │
│  │       │              (lineups, features, scoring)         │      │
│  │       ▼                                                   │      │
│  │  /backtest ──► backtester.py ──► streak simulation        │      │
│  │                                                           │      │
│  │  /train/tuned ──► hyperparameter_tuning.py                │      │
│  │                   (RandomizedSearchCV, 60 iter x 5 fold)  │      │
│  │                                                           │      │
│  │  /shap ──► shap_analysis.py ──► feature importance        │      │
│  └───────────────────────────────────────────────────────────┘      │
│                          │                                          │
└──────────────────────────│──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AWS SERVICES                                │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐     │
│  │  EventBridge  │  │              S3 Bucket                   │     │
│  │              │  │                                          │     │
│  │  Morning     │  │  predictions/{date}/top10_picks.csv      │     │
│  │  (11:30 ET)  │──│  predictions/{date}/all_predictions.csv  │     │
│  │  Evening     │  │  predictions/latest/                     │     │
│  │  (5:00 ET)   │  │  backtest/{season}/{timestamp}/          │     │
│  │      │       │  │  tuning/{timestamp}/                     │     │
│  │      ▼       │  │  analysis/shap/{timestamp}/              │     │
│  │   Lambda     │  │  models/hit_predictor.pkl                │     │
│  │  (trigger)   │  │  lambda/prediction_trigger.zip           │     │
│  └──────────────┘  └──────────────────┬───────────────────────┘     │
│                                       │                             │
│                                       ▼                             │
│                    ┌──────────────────────────────────────────┐     │
│                    │         Amazon QuickSight                 │     │
│                    │                                          │     │
│                    │  Data Source: S3 (predictions/latest/)   │     │
│                    │                                          │     │
│                    │  Dashboard:                              │     │
│                    │    • Top 10 Daily Picks (table)          │     │
│                    │    • Confidence Distribution (histogram) │     │
│                    │    • Backtest Performance (line chart)   │     │
│                    │    • Feature Importance (bar chart)      │     │
│                    │    • Streak Tracker                      │     │
│                    └──────────────────────────────────────────┘     │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  CloudWatch   │                                                   │
│  │  Logs         │  /ecs/test, /ecs/prod                            │
│  │  (monitoring) │  grep: [PREDICT], [TUNE], [BACKTEST], [SHAP]     │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Map

```
main.py (FastAPI entrypoint, v1.2)
├── historical_data.py      ← MLB Stats API data pulls, schedule enrichment
│   └── (MLB Stats API)        pitcher ERA, WHIP, K/9, park factors per game
├── feature_engineering.py  ← 26-feature pipeline, rolling stats, AABBH
│   └── era_tier_features.py   ← ERA tier classification + batter-vs-tier stats
├── model.py                ← XGBoost training, calibration, save/load
├── daily_predictions.py    ← Live prediction pipeline (lineups → features → rank)
├── backtester.py           ← Historical streak simulation, S3 result storage
├── hyperparameter_tuning.py← RandomizedSearchCV optimization
├── shap_analysis.py        ← SHAP explainability, miss analysis
└── lambda/
    └── prediction_trigger.py ← EventBridge → Lambda → Fargate trigger (prod only)
```

### Data Flow (Training)

1. `historical_data.py` pulls game logs from MLB Stats API (2000-2024)
2. Each game row is enriched with opposing pitcher stats (ERA, WHIP, K/9, handedness) and park factors
3. `feature_engineering.py` computes 26 rolling/cumulative features per batter-game row
4. `model.py` trains XGBoost with sample weights (recent seasons weighted higher)
5. Model is calibrated via `CalibratedClassifierCV` for reliable probability outputs
6. Trained model saved to `/tmp/models/hit_predictor.pkl`

### Data Flow (Daily Predictions)

1. `daily_predictions.py` pulls today's scheduled games + confirmed starting lineups
2. For each confirmed starter: pulls game log, computes all 26 features
3. Scores every batter through the trained model → probability of getting a hit
4. Ranks by confidence, outputs top 10 picks
5. Writes results to S3 for QuickSight dashboard consumption

### Data Flow (Backtest)

1. `backtester.py` iterates over game dates in a historical season
2. For each date: runs the full prediction pipeline as if it were "today"
3. Checks actual game results to verify if the top pick got a hit
4. Computes accuracy rates, longest streaks, monthly breakdown
5. Writes detailed results to S3 (summary.csv, daily_results.csv, full_results.json)

---

## 4. Feature Engineering

The model uses **26 features** across five categories. All rolling/cumulative features use `shift(1)` to prevent data leakage (only prior games are used).

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
| `games_between_hits` | Consistency metric (games played / hitless games + 1) |
| `hit_streak` | Current consecutive games with a hit |
| `game_num_in_season` | How deep into the season (handles early-season noise) |

### AABBH Features (4) — Patent Pending

The core innovation. AABBH (Average At Bats Between Hits) captures a batter's natural hitting rhythm and identifies when they're statistically due for a hit.

| Feature | Description |
|---------|-------------|
| `aabbh` | Total ABs / Total Hits (lower = more frequent hitter) |
| `current_hitless_abs` | Consecutive ABs since last hit |
| `aabbh_ratio` | `current_hitless_abs / aabbh` — the "overdue" signal |
| `in_optimal_zone` | 1 if `aabbh_ratio` is 0.5–1.0 (81.8% hit rate in backtest) |

The optimal zone (0.5–1.0 AABBH ratio) represents the "due but not overdue" sweet spot, where the batter's current hitless drought is between half and equal to their natural rhythm.

### ERA Tier Features (3) — v1.2

Captures how a batter performs against different quality tiers of pitching.

| Feature | Description |
|---------|-------------|
| `batter_hit_rate_vs_era_tier` | Hit rate against today's pitcher's ERA tier |
| `batter_ab_vs_era_tier` | Sample size for that tier (model learns reliability) |
| `era_tier_delta` | Tier-specific rate minus overall rate (specialist signal) |

**ERA Tiers:** Elite (<3.00), Good (3.00–3.75), Average (3.75–4.50), Below Average (4.50+)

Positive `era_tier_delta` means the batter performs better against this tier than their overall average. The model uses `batter_ab_vs_era_tier` to weight reliability — a tier rate based on 5 ABs is treated differently than one based on 80 ABs.

### Matchup Features (2)

| Feature | Description |
|---------|-------------|
| `is_home` | Home field advantage (1/0) |
| `platoon_advantage` | 1 if batter has handedness advantage vs pitcher |

### Pitcher Features (4)

| Feature | Description |
|---------|-------------|
| `pitcher_era` | Opposing starter's season ERA |
| `pitcher_whip` | Opposing starter's WHIP |
| `pitcher_k_per_9` | Opposing starter's K/9 |
| `pitcher_hand_encoded` | Pitcher handedness: R=0, L=1 |

### Environment Features (1)

| Feature | Description |
|---------|-------------|
| `park_factor` | Venue run factor (>1.0 = hitter-friendly) |

### Training Weight

Recency weighting ensures the model prioritizes modern baseball:

| Seasons | Weight |
|---------|--------|
| 2020–2024 | 1.0 |
| 2015–2019 | 0.7 |
| 2010–2014 | 0.4 |
| 2000–2009 | 0.2 |

---

## 5. Model Output & Usage

### Daily Predictions

The `/predict/daily` endpoint produces a ranked list of every confirmed starter, scored by probability of getting at least one hit. Output includes:

```json
{
  "date": "2026-04-15",
  "total_batters_scored": 142,
  "top_10_picks": [
    {
      "rank": 1,
      "player": "Player Name",
      "team": "Team Name",
      "opponent": "Opponent",
      "confidence": 82.4,
      "batting_avg": 0.312,
      "aabbh_ratio": 0.73,
      "in_optimal_zone": 1,
      "pitcher_era": 4.85,
      "platoon_adv": 1
    }
  ]
}
```

### How to Use the Picks

For **Beat the Streak**: Pick the #1 ranked player each day. The model achieved a 78.9% hit rate on top-1 picks in out-of-sample testing (2024 season), with a longest streak of 21 games.

For **prediction markets** (Kalshi, etc.): Use the confidence score as a probability estimate. The model is calibrated, meaning an 80% confidence prediction should result in a hit approximately 80% of the time.

### S3 Output Structure

```
s3://mlb-hit-predictor-{env}-bucket/
├── predictions/
│   ├── {date}/
│   │   ├── all_predictions.csv      (full ranked list, all starters)
│   │   └── top10_picks.csv          (top 10 with key features)
│   └── latest/
│       ├── all_predictions.csv      (always points to most recent)
│       └── top10_picks.csv
├── backtest/
│   └── {season}/{timestamp}/
│       ├── summary.csv              (headline metrics)
│       ├── daily_results.csv        (hit/miss per date)
│       └── full_results.json        (complete detail with features)
├── tuning/
│   └── {timestamp}/
│       └── tuning_results.json      (best params, CV scores, top combos)
├── analysis/
│   └── shap/{timestamp}/
│       └── shap_analysis.json       (feature importance, miss analysis)
└── models/
    └── hit_predictor.pkl            (trained model artifact)
├── lambda/
│   └── prediction_trigger.zip       (Lambda deployment package)
```

### QuickSight Dashboard

QuickSight connects to the S3 bucket and reads from `predictions/latest/` for a live daily dashboard:

- **Top 10 Picks Table**: Ranked picks with confidence, batting avg, AABBH ratio, pitcher ERA
- **Confidence Distribution**: Histogram of all scored batters
- **Backtest Performance**: Time series of hit rate by month
- **Feature Importance**: SHAP-based bar chart of which features drive predictions
- **Streak Tracker**: Running streak length during live season play

---

## 6. API Endpoints

| Method | Endpoint | Description | Runtime |
|--------|----------|-------------|---------|
| `GET` | `/` | API info + endpoint list | Instant |
| `GET` | `/health` | Health check (ECS uses this) | Instant |
| `POST` | `/train` | Train on single season | 30–60 min |
| `POST` | `/train/full` | Train on multiple seasons (2000–2024) | 2–4 hours |
| `POST` | `/train/tuned` | Hyperparameter tuning (RandomizedSearchCV) | 2–4 hours |
| `GET` | `/predict/daily` | Today's ranked predictions | 5–10 min |
| `POST` | `/backtest` | Run streak simulation on a season | 3–5 hours |
| `POST` | `/backtest-async` | Background backtest (results to S3) | 3–5 hours |
| `POST` | `/shap` | SHAP feature importance analysis | 5–15 min |
| `POST` | `/run-script` | Execute ad-hoc script from S3 | Varies |

### Key Parameters

**Training:**
```bash
POST /train/full?start_year=2021&end_year=2023
POST /train/tuned?start_year=2021&end_year=2023&n_iter=60&cv_folds=5
```

**Predictions:**
```bash
GET /predict/daily?date=2026-04-15&season=2026
```

**Backtesting:**
```bash
POST /backtest?season=2024&interval=2
# interval=2 means every other game date (~90 dates)
```

---

## 7. Monitoring & Operations

### Log Prefixes

Each module uses a distinct log prefix for easy filtering:

| Prefix | Module | Example grep |
|--------|--------|-------------|
| `[PREDICT]` | daily_predictions.py | `grep "PREDICT"` |
| `[TUNE]` | hyperparameter_tuning.py | `grep "TUNE"` |
| `[BACKTEST]` | backtester.py | `grep "BACKTEST"` |
| `[SHAP]` | shap_analysis.py | `grep "SHAP"` |

### Common Operations

```bash
# Check container health
curl -s http://$PUBLIC_IP:8080/health

# Tail live logs
aws logs tail /ecs/test --since 5m --no-paginate

# Check if long-running job is done
aws logs tail /ecs/test --since 8h --no-paginate | grep "COMPLETE"

# Pull latest backtest results
aws s3 ls s3://mlb-hit-predictor-test-bucket/backtest/2024/ --recursive | sort | tail -5

# Rollback to default model (if tuning made things worse)
curl -X POST "http://$PUBLIC_IP:8080/train/full?start_year=2021&end_year=2023"
```

### EventBridge Schedule (Production Only)

Automated daily predictions are triggered via EventBridge → Lambda → Fargate. The Lambda function discovers the running task's public IP, checks the MLB schedule for games, and hits `/predict/daily`. On off-days, it writes a no-games marker to S3 so QuickSight shows current status instead of stale data.

Two triggers run during the MLB season (March–October):

| Trigger | Schedule (ET) | Purpose |
|---------|--------------|---------|
| **Morning** | 11:30 AM | Day game lineups — lock early picks or evaluate the full slate |
| **Evening** | 5:00 PM | Night game lineups — updated pool with confirmed starters |

Both triggers are gated behind `EnableScheduledPredictions=true` and `Environment=production`. The test environment never creates scheduling resources.

**BTS Strategy:** Each prediction run scores every confirmed starter independently. Confidence scores are per-batter probabilities, not relative rankings — so a partial pool (morning) and full pool (evening) produce the same scores for the same players. Pick the highest confidence player from whichever run has the most confirmed lineups.

### Setting the Public IP

The ECS Fargate task receives a new public IP on each deployment. To retrieve it:

```bash
# Set your cluster name
CLUSTER=test-cluster

TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER --query 'taskArns[0]' --output text)

PUBLIC_IP=$(aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text | xargs -I {} aws ec2 describe-network-interfaces \
  --network-interface-ids {} \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo $PUBLIC_IP
```

> **Note:** This variable is session-scoped. If you get empty responses from `curl`, re-run the above commands.

---

## Version History

| Version | Date | Changes | OOS Result |
|---------|------|---------|------------|
| v0.1.0 | 2026-01-15 | Initial model with basic batting features | — |
| v0.2.0 | 2026-02-01 | AABBH features, optimal zone discovery | — |
| v0.3.0 | 2026-02-05 | SHAP analysis, async backtest, S3 integration | — |
| v0.4.0 | 2026-02-10 | Hyperparameter tuning endpoint | — |
| v1.0.0 | 2026-02-12 | Baseline: 23 features, default XGBoost params | 78.4% in-sample |
| v1.1.0 | 2026-02-14 | Lineup pre-check, doubleheader handling, prior-season AABBH fallback | **76.7%** (69/90), 13-game streak |
| v1.2.0 | 2026-02-15 | ERA tier features (26 features), pickle save bug fix, calibration fix, EventBridge + Lambda automation, CI/CD hardcoded values removed, deploy workflow cleanup | **78.9%** (71/90), **21-game streak** |