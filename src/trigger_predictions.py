import requests
import time

sample_payload = {
    "features": ["Riverbend Stake", "Londonderry", 3, 2, 1.5, 30, "No", "2", 50]
}

# Wait for API to be up
time.sleep(10)

print("Triggering predictions to populate Grafana...")

for i in range(5):
    try:
        r = requests.post("http://localhost:9999/v1/predict", json=sample_payload)
        print(f"[{i+1}] Response:", r.json())
    except Exception as e:
        print("Prediction error:", e)
    time.sleep(2)
