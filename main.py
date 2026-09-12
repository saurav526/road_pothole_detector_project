import requests
import pandas as pd

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 18.52,
    "longitude": 73.85,
    "current": "temperature_2m,wind_speed_10m"
}

response = requests.get(url, params=params)

data = response.json()
print(data)

df = pd.DataFrame([data["current"]])

print(df)

print(data["latitude"])
print(data["longitude"])
print(data["current"]["temperature_2m"])
print(data["current"]["wind_speed_10m"])


try:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    print(data)

