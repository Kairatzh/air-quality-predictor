import pandas as pds
import numpy as nmp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

""" --Import Data-- """
data = pds.read_csv("C:/Users/User/Desktop/projects/air-quality-predictor/data/airPollutions.csv")

""" --Check Data-- """
print(data.head(3))
print(data.info())
print(data.describe())
print(data.shape)

""" --Data Engineering-- """
cols = [
    "Temperature", "Humidity", "PM2.5", "PM10",
    "NO2", "SO2", "CO",
    "Proximity_to_Industrial_Areas",
    "Population_Density"
]

data = data.drop_duplicates()
data[cols] = data[cols].fillna(data[cols].mean())
data = data.dropna(subset=cols)
data["Air Quality"] = data["Air Quality"].fillna(data["Air Quality"].mode()[0])

""" -- Final data-- """
print(data.head(3))
print(data.info())
print(data.describe())
print(data.shape)


""" --Scaling Data-- """
scaler = StandardScaler()
X = data.drop(columns=["Air Quality"])
y = data["Air Quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
