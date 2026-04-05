from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uvicorn
import os
import sys

# Adjust path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.predict import predict_intent

# Initialize FastAPI App
app = FastAPI(
    title="IntentIQ: LeadIntent AI",
    description="Production-grade intent classification for CRM-based sales interactions.",
    version="1.0.0"
)

# Request Data Model
class IntentRequest(BaseModel):
    text: str = Field(..., min_length=3, json_schema_extra={"example": "Need pricing and demo"})

# Response Data Model
class IntentResponse(BaseModel):
    intent: str
    confidence: float
    probabilities: dict

@app.post("/predict-intent", response_model=IntentResponse)
async def predict_endpoint(request: IntentRequest):
    """
    Classify input text based on customer interaction notes into:
    High Intent, Medium Intent, Low Intent, Inquiry, Complaint.
    """
    try:
        # Logging for prediction
        print(f"[PREDICT] Request Input: '{request.text}'")
        
        # Core prediction logic
        result = predict_intent(request.text)
        
        # Log prediction result
        print(f"[RESULT] Intent: {result['intent']} | Confidence: {result['confidence']:.4f}")
        
        return IntentResponse(
            intent=result['intent'],
            confidence=result['confidence'],
            probabilities=result['probabilities']
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Model or vectorizer not found: {e}")
        raise HTTPException(status_code=500, detail="Intent model not found. Please ensure the model is trained.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.get("/")
async def root():
    return {"status": "online", "message": "IntentIQ API is active and ready for predictions."}

if __name__ == "__main__":
    # Note: Fast API typically runs via uvicorn in terminal
    # uvicorn api.app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
