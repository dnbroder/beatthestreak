"""
model_retrainer.py – Lambda function triggered by EventBridge (annually).

Automatically kicks off model retraining on the most recent 3 seasons.
Since training + validation takes 2+ hours (exceeding Lambda's 15-min limit),
this Lambda fires the request and returns immediately. The ECS container
handles training, validation, and S3 writes.

The /train/retrain-auto endpoint (added in main.py) runs the full pipeline:
  1. Train on current_year-3 to current_year-1
  2. Backtest on current_year-1 (interval=3)
  3. Write results to S3
  4. Send SNS notification with results

EventBridge schedule: February 15 at 10:00 AM ET (pre-season)
"""

import boto3
import json
import urllib.request
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_task_public_ip(cluster):
    """Discover the public IP of the running Fargate task."""
    ecs = boto3.client("ecs")
    ec2 = boto3.client("ec2")

    tasks = ecs.list_tasks(cluster=cluster, desiredStatus="RUNNING")
    task_arns = tasks.get("taskArns", [])
    if not task_arns:
        raise Exception(f"No running tasks in cluster {cluster}")

    task_detail = ecs.describe_tasks(cluster=cluster, tasks=[task_arns[0]])
    attachments = task_detail["tasks"][0].get("attachments", [])

    eni_id = None
    for attachment in attachments:
        for detail in attachment.get("details", []):
            if detail.get("name") == "networkInterfaceId":
                eni_id = detail["value"]
                break

    if not eni_id:
        raise Exception("Could not find ENI for task")

    eni = ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
    public_ip = eni["NetworkInterfaces"][0]["Association"]["PublicIp"]
    return public_ip


def send_notification(topic_arn, subject, message):
    """Send SNS notification."""
    if not topic_arn:
        logger.warning("[RETRAIN] No SNS topic configured, skipping notification")
        return
    try:
        sns = boto3.client("sns")
        sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)
        logger.info(f"[RETRAIN] SNS notification sent: {subject}")
    except Exception as e:
        logger.error(f"[RETRAIN] Failed to send SNS notification: {e}")


def handler(event, context):
    cluster = os.environ["ECS_CLUSTER"]
    port = os.environ["CONTAINER_PORT"]
    env = os.environ.get("ENVIRONMENT", "production")
    sns_topic = os.environ.get("SNS_TOPIC_ARN", "")

    current_year = datetime.now(timezone.utc).year
    end_year = current_year - 1
    start_year = end_year - 2  # 3-year window

    logger.info(f"[RETRAIN] ============================================================")
    logger.info(f"[RETRAIN] ANNUAL MODEL RETRAIN — {current_year} Pre-Season")
    logger.info(f"[RETRAIN] Training: {start_year}-{end_year}")
    logger.info(f"[RETRAIN] Environment: {env}")
    logger.info(f"[RETRAIN] ============================================================")

    try:
        # Discover task IP
        ip = get_task_public_ip(cluster)
        base_url = f"http://{ip}:{port}"
        logger.info(f"[RETRAIN] Found task at {base_url}")

        # Health check
        health_resp = urllib.request.urlopen(f"{base_url}/health", timeout=15)
        health_data = json.loads(health_resp.read().decode())
        logger.info(f"[RETRAIN] Health check: {health_data}")

        # Fire async retrain endpoint
        # This endpoint runs training + backtest + SNS notification in a background thread
        retrain_url = (
            f"{base_url}/train/retrain-auto"
            f"?start_year={start_year}&end_year={end_year}"
            f"&validation_season={end_year}&sns_topic={sns_topic}"
        )
        req = urllib.request.Request(retrain_url, method="POST")
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())

        logger.info(f"[RETRAIN] Async retrain triggered: {result}")

        send_notification(
            sns_topic,
            f"[RETRAIN STARTED] {current_year} Pre-Season",
            f"Annual model retrain has been kicked off.\n\n"
            f"Training: {start_year}-{end_year}\n"
            f"Validation: {end_year} season (interval=3)\n\n"
            f"This will take 2-3 hours. You will receive another notification\n"
            f"when training and validation are complete with results.",
        )

        return {
            "statusCode": 200,
            "status": "triggered",
            "training": f"{start_year}-{end_year}",
            "message": "Retrain pipeline started on ECS. Results via SNS.",
        }

    except Exception as e:
        error_msg = f"Retrain trigger failed: {str(e)}"
        logger.error(f"[RETRAIN] {error_msg}")
        send_notification(
            sns_topic,
            f"[RETRAIN FAILED] {current_year} Pre-Season — Trigger Error",
            f"Could not start annual retrain.\n\nError: {str(e)}\n\n"
            f"Check: Is the ECS container running? Is the IP correct?\n"
            f"Manual intervention required.",
        )
        raise