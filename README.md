# IntentIQ: LeadIntent AI System

**IntentIQ** is a production-grade NLP classification system designed to categorize customer interactions into actionable intent categories. Built for performance and reliability, it classifies customer messages into five key intents to help sales teams prioritize high-value leads automatically.

## Intent Categories
- **High Intent**: Immediate interest, pricing/demo requests, ready to buy.
- **Medium Intent**: Evaluating options, follow-up later, internal review.
- **Low Intent**: Just browsing, not interested, wrong contact.
- **Inquiry**: General questions about features, integrations, or security.
- **Complaint**: Technical issues, billing problems, or poor experience.

## Real-World Impact
This model is designed to integrate with **CRM systems** (like Salesforce or HubSpot) to analyze sales interaction notes in real-time. By automatically identifying **High Intent** leads, organizations can ensure faster response times and higher conversion rates, transforming raw data into prioritized sales opportunities.

---

## Project Structure
```text
IntentIQ/
├── api/                # FastAPI application layer
│   └── app.py
├── data/               # Curated training data
│   └── dataset.csv
├── model/              # ML training and prediction logic
│   ├── model_v1.pkl
│   ├── vectorizer_v1.pkl
│   ├── train.py
│   └── predict.py
├── notebook/           # Analytical experimentation
│   └── experiments.ipynb
├── utils/              # Data processing utilities
│   └── preprocess.py
├── requirements.txt    # Project dependencies
└── README.md
```

## Tech Stack
- **Language**: Python 3.x
- **Frameworks**: FastAPI, Scikit-learn, Pandas, Joblib
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook

---

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
This will evaluate the dataset, generate the versioned artifacts, and save a confusion matrix visualization.
```bash
python model/train.py
```

### 3. Run the API
```bash
uvicorn api.app:app --reload
```

### 4. API Endpoints
- **POST /predict-intent**:
  - Request: `{ "text": "Need demo and pricing ASAP" }`
  - Response:
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

## Model Evaluation
The system uses **Logistic Regression** chosen for its native probability scoring and interpretability. On the curated dataset of 250+ rows, it achieves high precision across all classes. A visual confusion matrix (`model/confusion_matrix.png`) is generated during training to provide deeper insights into model performance.

---

## Authors
**LeadIntent AI Team** - *Empowering Sales through Applied Intelligence*
