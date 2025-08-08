import json
from datetime import datetime

def log_request(inputs, prediction):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": inputs,
        "prediction": prediction
    }
    with open("log/predictions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
