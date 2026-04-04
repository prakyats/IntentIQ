import os
import joblib
import numpy as np
from typing import Dict, Any

# Adjust paths for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text

# Load model artifacts (using versioning _v1)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_v1.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer_v1.pkl")

# Lazy loading of model and vectorizer
_model = None
_vectorizer = None

def load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Model artifacts not found. Please run train.py first.")
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)

def predict_intent(text: str) -> Dict[str, Any]:
    """
    Predicts the intent of the given text.
    Handles edge cases like very long text and mixed-intent sentences.
    Returns:
        Dict: { 'intent': str, 'confidence': float, 'probabilities': Dict[str, float] }
    """
    # Check for empty or very short input which shouldn't happen after API validation but good for robustness
    if not text or len(text.strip()) < 3:
        return {
            "intent": "Unknown",
            "confidence": 1.0,
            "probabilities": {}
        }
    
    load_artifacts()
    
    # Preprocess text
    cleaned_text = clean_text(text)
    
    # Edge Case: Very long text (Optional: Truncate to first 1000 chars to avoid vectorization bottlenecks)
    if len(cleaned_text) > 1000:
        cleaned_text = cleaned_text[:1000]
    
    # Vectorize
    vectorized_text = _vectorizer.transform([cleaned_text])
    
    # Get class probabilities
    probs = _model.predict_proba(vectorized_text)[0]
    classes = _model.classes_
    
    # Get the best intent and its confidence
    best_index = np.argmax(probs)
    best_intent = classes[best_index]
    confidence = float(np.round(probs[best_index], 4))
    
    # Create full probabilities dictionary
    full_probs = {cls: float(np.round(prob, 4)) for cls, prob in zip(classes, probs)}
    
    return {
        "intent": best_intent,
        "confidence": confidence,
        "probabilities": full_probs
    }

if __name__ == "__main__":
    # Test cases
    test_inputs = [
        "Need pricing details urgently",
        "Interesting tool but we don't have the budget right now, maybe later?",
        "Facing issue with the login dashboard, it keeps crashing my browser",
        "hi", # Very short text
        "Just checking out your website and looking for features list"
    ]
    
    print("Testing Predictions...")
    for text in test_inputs:
        result = predict_intent(text)
        print(f"Input: '{text}'")
        print(f"Result: {result['intent']} (Conf: {result['confidence']})")
        print("-" * 30)
