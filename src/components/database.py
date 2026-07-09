import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "telemetry_db.json")

def init_db():
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, "w") as f:
            json.dump({"students": {}}, f)

def save_prediction(student_id, features, predicted_score):
    init_db()
    with open(DB_PATH, "r") as f:
        data = json.load(f)
        
    if student_id not in data["students"]:
        data["students"][student_id] = []
        
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_score": float(predicted_score),
        "features": features
    }
    
    data["students"][student_id].append(entry)
    
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)

def get_student_history(student_id):
    init_db()
    with open(DB_PATH, "r") as f:
        data = json.load(f)
    
    return data["students"].get(student_id, [])
