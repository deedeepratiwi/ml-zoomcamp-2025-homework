import pickle
from fastapi import FastAPI
import uvicorn

from typing import Dict, Any

app = FastAPI(title="customer-conversion")

print("Loading model...")
with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)
print("✅ Model loaded successfully")

def predict_single(customer):
    result = pipeline.predict_proba([customer])[0, 1]
    print("Prediction computed:", result)
    return float(result)


@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(customer: dict):
    print("Received request:", customer)
    prob = predict_single(customer)
    print("Returning response")

    return {
        "convert_probability": prob,
        "convert": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)