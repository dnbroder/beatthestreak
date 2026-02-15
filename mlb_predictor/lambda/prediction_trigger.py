"""
prediction_trigger.py – Lambda function triggered by EventBridge.

Discovers the Fargate task IP, checks if games are scheduled,
and hits /predict/daily on the running ECS container.

Two triggers per day (production only):
  - Morning (11:30 AM ET): Day game lineups
  - Evening (5:00 PM ET): Night game lineups
"""

import boto3
import json
import urllib.request
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&date={date}"


def check_games_today(date_str):
    """Check MLB schedule API for games today. Returns game count."""
    try:
        url = MLB_SCHEDULE_URL.format(date=date_str)
        resp = urllib.request.urlopen(url, timeout=10)
        data = json.loads(resp.read().decode())
        total = data.get('totalGames', 0)
        logger.info(f"MLB schedule for {date_str}: {total} games")
        return total
    except Exception as e:
        logger.warning(f"Could not check MLB schedule: {e}")
        # If schedule check fails, proceed anyway (fail open)
        return -1


def write_no_games_marker(date_str, env):
    """Write a marker to S3 so QuickSight shows 'No games' instead of stale data."""
    try:
        s3 = boto3.client('s3')
        bucket = f"mlb-hit-predictor-{env}-bucket"
        marker = {
            "date": date_str,
            "status": "no_games_scheduled",
            "message": "No MLB games scheduled for this date"
        }
        # Update latest so QuickSight doesn't show yesterday's picks
        s3.put_object(
            Bucket=bucket,
            Key=f"predictions/{date_str}/status.json",
            Body=json.dumps(marker),
            ContentType="application/json"
        )
        s3.put_object(
            Bucket=bucket,
            Key="predictions/latest/status.json",
            Body=json.dumps(marker),
            ContentType="application/json"
        )
        logger.info(f"No-games marker written to S3 for {date_str}")
    except Exception as e:
        logger.warning(f"Could not write no-games marker: {e}")


def get_task_public_ip(cluster):
    """Discover the public IP of the running Fargate task."""
    ecs = boto3.client('ecs')
    ec2 = boto3.client('ec2')

    # Get running task ARN
    tasks = ecs.list_tasks(cluster=cluster, desiredStatus='RUNNING')
    task_arns = tasks.get('taskArns', [])
    if not task_arns:
        raise Exception(f"No running tasks in cluster {cluster}")

    # Get task details to find ENI
    task_detail = ecs.describe_tasks(cluster=cluster, tasks=[task_arns[0]])
    attachments = task_detail['tasks'][0].get('attachments', [])

    eni_id = None
    for attachment in attachments:
        for detail in attachment.get('details', []):
            if detail.get('name') == 'networkInterfaceId':
                eni_id = detail['value']
                break

    if not eni_id:
        raise Exception("Could not find ENI for task")

    # Get public IP from ENI
    eni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
    public_ip = eni['NetworkInterfaces'][0]['Association']['PublicIp']
    return public_ip


def handler(event, context):
    cluster = os.environ['ECS_CLUSTER']
    port = os.environ['CONTAINER_PORT']
    env = os.environ.get('ENVIRONMENT', 'production')
    trigger = event.get('trigger', 'morning')

    # Today's date
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    logger.info(f"Prediction trigger fired: {trigger} run for {today}")

    # ---- Game-day check ----
    game_count = check_games_today(today)
    if game_count == 0:
        logger.info(f"No games today ({today}). Writing marker and skipping.")
        write_no_games_marker(today, env)
        return {
            'statusCode': 200,
            'trigger': trigger,
            'date': today,
            'message': 'No games scheduled, skipped prediction run'
        }

    try:
        # Discover task IP
        ip = get_task_public_ip(cluster)
        logger.info(f"Found task at {ip}:{port}")

        # Health check first
        health_url = f"http://{ip}:{port}/health"
        health_req = urllib.request.urlopen(health_url, timeout=10)
        logger.info(f"Health check: {health_req.read().decode()}")

        # Trigger predictions
        predict_url = f"http://{ip}:{port}/predict/daily"
        req = urllib.request.Request(predict_url, method='GET')
        response = urllib.request.urlopen(req, timeout=600)
        result = json.loads(response.read().decode())

        total = result.get('total_batters_scored', 0)
        top_pick = result.get('top_pick', {})
        logger.info(f"Predictions complete: {total} batters scored")
        logger.info(f"Top pick: {top_pick}")

        return {
            'statusCode': 200,
            'trigger': trigger,
            'date': today,
            'batters_scored': total,
            'top_pick': top_pick
        }

    except Exception as e:
        logger.error(f"Prediction trigger failed: {str(e)}")
        raise