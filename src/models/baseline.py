import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.feature_engineering import tfidf_vectorize

# Định nghĩa đường dẫn dữ liệu
DATA_DIR = "data/processed"
RESULTS_DIR = "results/tables"
MODEL_DIR = "models/baseline"

# Tạo thư mục nếu chưa có
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Đọc dữ liệu từ các file có sẵn
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Vector hóa văn bản bằng TF-IDF (dùng tập train để fit vectorizer)
X_train, vectorizer = tfidf_vectorize(train_df['clean_text'])
y_train = train_df['label']

# Biến đổi tập test theo vectorizer đã fit
X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['label']

# Danh sách mô hình baseline
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
}

# GridSearchCV: Danh sách tham số cho từng mô hình
param_grids = {
    "Naive Bayes": {"alpha": [0.1, 0.5, 1.0, 2.0]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
    "KNN": {"n_neighbors": [3, 5, 7, 9]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
}

# Lưu kết quả đánh giá
results = []

# Huấn luyện và đánh giá từng mô hình (Baseline + GridSearchCV)
for name, model in models.items():
    print(f"Training {name} (Baseline)...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Lưu kết quả baseline
    results.append({
        "Model": name,
        "Type": "Baseline",
        "Accuracy": acc,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score']
    })

    # Lưu mô hình baseline
    model_filename = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_baseline.pkl")
    joblib.dump(model, model_filename)

    print(f"Training {name} (GridSearchCV)...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred_best = best_model.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred_best)
    best_report = classification_report(y_test, y_pred_best, output_dict=True)

    # Lưu kết quả GridSearchCV
    results.append({
        "Model": name,
        "Type": "GridSearchCV",
        "Accuracy": best_acc,
        "Precision": best_report['weighted avg']['precision'],
        "Recall": best_report['weighted avg']['recall'],
        "F1-Score": best_report['weighted avg']['f1-score']
    })

    # Lưu mô hình tối ưu từ GridSearchCV
    best_model_filename = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_gridsearch.pkl")
    joblib.dump(best_model, best_model_filename)

    print(f"{name} - Best Params: {best_params}")
    print(f"Baseline Accuracy: {acc:.4f} | GridSearchCV Accuracy: {best_acc:.4f}")
    print("=" * 60)

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(results)

# Lưu kết quả vào results/tables/
results_file = os.path.join(RESULTS_DIR, "baseline_results.csv")
results_df.to_csv(results_file, index=False)
print(f"Baseline results saved to: {results_file}")
