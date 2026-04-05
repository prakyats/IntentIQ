import requests
import json

# Define the API endpoint
url = "http://127.0.0.1:8000/predict-intent"

# Sample inputs to test different intent categories
samples = [
    "Need pricing and demo ASAP",
    "Not interested right now, maybe next year",
    "Dashboard is crashing when I save the lead details",
    "What are your security features and integrations?",
    "Trying the tool but still not sure about it, let's talk in a week"
]

print("="*60)
print(" IntentIQ: LeadIntent AI Live Demo ".center(60, "="))
print("="*60)

for i, text in enumerate(samples, 1):
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"\n[{i}] Input: \"{text}\"")
            print(f"    - Predicted Intent: {result['intent']}")
            print(f"    - Confidence: {result['confidence'] * 100:.2f}%")
            print(f"    - Full distribution: {json.dumps(result['probabilities'], indent=8)}")
        else:
            print(f"\n[!] Error with sample {i}: API returned status code {response.status_code}")
    except Exception as e:
        print(f"\n[!] Connection Error: {e}")

print("\n" + "="*60)
print(" Demo Complete ".center(60, "="))
print("="*60)
