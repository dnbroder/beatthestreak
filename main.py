from fastapi import FastAPI, UploadFile, File
from mlb_predictor.data_loader import load_data
from mlb_predictor.predictor import Predictor
import pandas as pd
import io

app = FastAPI()
predictor = Predictor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    results = predictor.predict(df)
    return results.to_dict(orient="records")
