# 🚀 IntentIQ — LeadIntent AI

**IntentIQ** is a production-grade NLP system that classifies customer interaction text into actionable intent categories, enabling intelligent lead prioritization in CRM systems.

Transform unstructured sales conversations into clear, data-driven decisions.

---

## 🧠 Overview

Sales teams deal with large volumes of unstructured data such as call notes and customer messages. Identifying high-value leads manually is inefficient and error-prone.

IntentIQ analyzes interaction text and automatically classifies it into meaningful intent categories, helping teams focus on the most promising opportunities.

---

## ❗ Problem Statement

- Sales interactions are stored as unstructured text  
- Hard to identify which leads matter most  
- Manual prioritization leads to missed opportunities and slower response  

---

## 💡 Solution

IntentIQ:
- Processes customer interaction text  
- Extracts intent signals  
- Classifies into predefined categories  
- Provides confidence-based predictions  

---

## 🎯 Intent Categories

- High Intent → Ready to buy, demo/pricing requests  
- Medium Intent → Evaluating, follow-up later  
- Low Intent → Browsing, low interest  
- Inquiry → General questions (features, integrations)  
- Complaint → Issues, dissatisfaction  

---

## ⚙️ Architecture

Input Text  
↓  
Preprocessing (cleaning, normalization)  
↓  
TF-IDF Vectorization  
↓  
Logistic Regression Model  
↓  
FastAPI Prediction  
↓  
JSON Output  

---

## 🤖 Model Details

- Algorithm: TF-IDF + Logistic Regression  
- Fast, lightweight, and interpretable  
- Works well on small-to-medium datasets  
- Provides probability-based confidence scores  

---

## 📊 Model Performance

- High accuracy across all classes  
- Balanced dataset ensures stable predictions  

Evaluation includes:
- Precision  
- Recall  
- F1-score  
- Confusion Matrix (model/confusion_matrix.png)  

---

## 🧪 Example Prediction

Input:
Need pricing details and a demo ASAP

Output:
```json
{
  "intent": "High Intent",
  "confidence": 0.9572,
  "probabilities": {
    "High Intent": 0.9572,
    "Medium Intent": 0.0210,
    "Low Intent": 0.0051,
    "Inquiry": 0.0123,
    "Complaint": 0.0044
  }
}
```

---

## 📦 Dataset

- 267 manually curated samples  
- Balanced across 5 intent classes  
- Simulates real CRM interaction notes  

Includes:
- informal phrasing  
- varied sentence lengths  
- realistic variations  

---

## 🔗 Real-World Impact

IntentIQ is designed to integrate with CRM systems.

Example:

Input:
Client asked for pricing and wants a demo next week  

Output:
High Intent  

Enables:
- Faster response to high-value leads  
- Better prioritization  
- Improved conversion rates  

---

## 📁 Project Structure

```
IntentIQ/
├── api/
│   └── app.py
├── data/
│   └── dataset.csv
├── model/
│   ├── model_v1.pkl
│   ├── vectorizer_v1.pkl
│   ├── train.py
│   └── predict.py
├── notebook/
│   └── experiments.ipynb
├── utils/
│   └── preprocess.py
├── requirements.txt
└── README.md
```

---

## ⚡ API Usage

POST /predict-intent

Request:
```json
{
  "text": "Need pricing"
}
```

Response:
```json
{
  "intent": "High Intent",
  "confidence": 0.92,
  "probabilities": {
    "High Intent": 0.92,
    "Medium Intent": 0.04,
    "Low Intent": 0.01,
    "Inquiry": 0.02,
    "Complaint": 0.01
  }
}
```

---

## 🛠️ Setup & Execution

Install dependencies:
```
pip install -r requirements.txt
```

Train model:
```
python model/train.py
```

Run API:
```
uvicorn api.app:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

---

## 🧪 Sample Inputs

- Need demo urgently  
- Just exploring options  
- Facing issue with product  
- Can you share pricing?  
- Not interested right now  

---

## ⚠️ Limitations

- May struggle with ambiguous or mixed-intent sentences  
- Performance depends on dataset quality  

---

## 🚀 Future Improvements

- BERT / transformer upgrade  
- Real-time CRM integration  
- Continuous learning  
- Lead scoring (intent + sentiment)  

---

## 👨‍💻 Authors

LeadIntent AI Team  
Empowering Sales through Applied Intelligence