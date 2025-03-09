import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
from src.feature_engineering import tfidf_vectorize

# Load dữ liệu đã xử lý
df = pd.read_csv("data/processed/20news_processed.csv")  

# Vector hóa văn bản bằng TF-IDF
X, vectorizer = tfidf_vectorize(df['clean_text'])
y = df['label']

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Danh sách mô hình baseline
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear'),
}

# Lưu kết quả đánh giá
results = []

# Huấn luyện và đánh giá từng mô hình
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Lưu kết quả
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score']
    })
    
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("="*50)

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)

# Tạo thư mục results nếu chưa có
os.makedirs("results", exist_ok=True)

# Lưu kết quả vào CSV
results_file = "results/baseline_results.csv"
results_df.to_csv(results_file, index=False)
print(f"Baseline results saved to: {results_file}")
