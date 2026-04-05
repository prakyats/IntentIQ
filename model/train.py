import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text

def train_model(data_path: str, model_save_path: str, vectorizer_save_path: str, confusion_matrix_path: str):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    
    X = df['text_cleaned']
    y = df['intent']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Text Vectorization
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Prediction and Evaluation
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 30)
    
    # Confusion Matrix Visualization
    print(f"Creating confusion matrix at {confusion_matrix_path}...")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('IntentIQ: Confusion Matrix (Portfolio Evaluation)')
    plt.xlabel('Predicted Intent')
    plt.ylabel('Actual Intent')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Save Model & Vectorizer (Versioning with _v1)
    print(f"Saving artifacts to {model_save_path} and {vectorizer_save_path}...")
    joblib.dump(model, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)
    
    print("Training complete.")

if __name__ == "__main__":
    # Define paths relative to this script's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
    MODEL_FILE = os.path.join(BASE_DIR, "model_v1.pkl")
    VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer_v1.pkl")
    PLOT_FILE = os.path.join(BASE_DIR, "confusion_matrix.png")
    
    train_model(DATA_PATH, MODEL_FILE, VECTORIZER_FILE, PLOT_FILE)
