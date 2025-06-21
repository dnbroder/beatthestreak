from fastapi import FastAPI, HTTPException
from mlb_predictor import data_loader, predictor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    logger.info("Health check endpoint hit.")
    return {"message": "MLB Predictor API is live."}

@app.get("/predict")
def run_prediction(source: str = "statsapi"):
    try:
        logger.info(f"Starting prediction using source: {source}")
        
        # Load data
        data = data_loader.load_data(source)
        if data.empty:
            logger.warning("Data loaded was empty.")
            raise HTTPException(status_code=500, detail="Failed to load data.")

        # Run prediction
        predictions = predictor.make_prediction(data)
        logger.info("Prediction completed successfully.")

        return {"predictions": predictions}
    
    except Exception as e:
        logger.exception("Prediction endpoint failed.")
        raise HTTPException(status_code=500, detail=str(e))
