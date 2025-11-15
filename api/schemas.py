from pydantic import BaseModel

class Request(BaseModel):
    Temperature: float
    Humidity: float
    PM25: float
    PM10: float
    NO2: float
    SO2: float
    CO: float
    Proximity_to_Industrial_Areas: float
    Population_Density: float

class Response(BaseModel):
    Air_Quality: str
