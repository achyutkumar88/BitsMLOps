import requests

url = "http://localhost:8000/predict"

payload = {
    "features": [
        [8.3252, 41, 6, 1.023809524, 322, 2.555555556, 37.88, -122.23]
    ]
}

response = requests.post(url, json=payload)
print(response.json())
