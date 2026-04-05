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
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

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
    
    # 1. Stratified Split (80/20)
    print("Performing stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Upgraded Feature Engineering (context aware)
    print("Vectorizing with TF-IDF (NGrams: 1-2)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 3. Model Comparison
    from sklearn.calibration import CalibratedClassifierCV
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', C=1.0),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=2000, dual=False))
    }
    
    best_model = None
    best_f1 = 0
    best_name = ""
    
    print("\n--- Model Comparison (F1-Score Selection) ---")
    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name}: Weighted F1 = {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
            best_name = name

    print(f"\n---> Selected Best Model: {best_name}")
    
    # 4. Detailed Evaluation of Best Model
    y_pred_best = best_model.predict(X_test_vec)
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_best))
    print("-" * 30)
    
    # 5. Mandatory Error Analysis (Top 10 Misclassified)
    print("\n--- Error Analysis (Top 10 Misclassified Samples) ---")
    test_results = pd.DataFrame({
        'text': X_test,
        'actual': y_test,
        'predicted': y_pred_best
    })
    misclassified = test_results[test_results['actual'] != test_results['predicted']]
    
    if len(misclassified) > 0:
        print(misclassified.head(10).to_string(index=False))
    else:
        print("No misclassifications found on test set! Check for overfitting.")
    print("-" * 30)
    
    # 6. Confusion Matrix Visualization
    print(f"Creating confusion matrix at {confusion_matrix_path}...")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred_best, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'IntentIQ: Confusion Matrix ({best_name})')
    plt.xlabel('Predicted Intent')
    plt.ylabel('Actual Intent')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # 7. Save Best Artifacts
    print(f"Saving artifacts to {model_save_path} and {vectorizer_save_path}...")
    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)
    
    print("Training complete.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
    MODEL_FILE = os.path.join(BASE_DIR, "model_v1.pkl")
    VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer_v1.pkl")
    PLOT_FILE = os.path.join(BASE_DIR, "confusion_matrix.png")
    
    train_model(DATA_PATH, MODEL_FILE, VECTORIZER_FILE, PLOT_FILE)
