from fastapi import APIRouter
from api.schemas import Request, Response
import joblib
import numpy as np

router = APIRouter()

model = joblib.load("C:/Users/User/Desktop/projects/air-quality-predictor/ml/model.joblib")

@router.post("/api/generate", response_model=Response)
def predict(request: Request):
    X = np.array([[request.Temperature,
                   request.Humidity,
                   request.PM25,
                   request.PM10,
                   request.NO2,
                   request.SO2,
                   request.CO,
                   request.Proximity_to_Industrial_Areas,
                   request.Population_Density]])

    y_pred = model.predict(X)
    return Response(Air_Quality=y_pred[0])
