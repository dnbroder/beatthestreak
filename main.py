from fastapi import FastAPI
from mlb_predictor.predictor import make_basic_prediction
import logging

# Configure logger to ensure visibility in CloudWatch
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 ECS container has started. Running initial prediction.")
    try:
        make_basic_prediction()
        logger.info("✅ Initial prediction completed successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to run prediction on startup: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the MLB predictor API. Use POST /predict to re-run predictions."}

@app.post("/predict")
def trigger_prediction():
    try:
        df = make_basic_prediction()
        return {
            "message": "✅ Prediction executed successfully.",
            "num_rows": len(df)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"message": "❌ Prediction failed.", "error": str(e)}
