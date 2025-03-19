import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.feature_engineering import tfidf_vectorize_ngrams

# Định nghĩa đường dẫn dữ liệu
DATA_DIR = "data/processed"
RESULTS_DIR = "results/tables"
MODEL_DIR = "models/deep_learning"

# Tạo thư mục nếu chưa có
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Đọc dữ liệu
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# Vector hóa văn bản bằng TF-IDF (dùng tập train để fit vectorizer)
X_train, vectorizer = tfidf_vectorize_ngrams(train_df['clean_text'])
y_train = train_df['label']

X_val = vectorizer.transform(val_df['clean_text'])
y_val = val_df['label']

X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['label']

# Chuyển đổi dữ liệu thành tensor
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int).values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(pd.to_numeric(y_val, errors='coerce').fillna(0).astype(int).values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(pd.to_numeric(y_test, errors='coerce').fillna(0).astype(int).values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mô hình MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=20):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Mô hình CNN
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=20, num_filters=100, kernel_size=3):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Thêm chiều kênh
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(2)
        x = self.fc(x)
        return x

# Mô hình RNN (LSTM)
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=20, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Thêm chiều seq_len=1
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return x

# Hàm huấn luyện
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Hàm đánh giá
def evaluate_model(model, test_loader, criterion):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())  # Move predictions to CPU
            true_labels.extend(y_batch.cpu().numpy())  # Move true labels to CPU

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)  # Handle zero division
    avg_loss = total_loss / len(test_loader)
    return accuracy, report, avg_loss

# Hàm lưu kết quả
def save_results(model_name, accuracy, report, results_dir):
    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score']
    }
    results_df = pd.DataFrame([results])
    results_file = os.path.join(results_dir, f"{model_name}_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"{model_name} results saved to: {results_file}")

# Hàm chính để huấn luyện và đánh giá mô hình
def train_and_evaluate(model, model_name, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training {model_name}...")
    train_model(model, train_loader, criterion, optimizer, epochs=num_epochs)

    print(f"Evaluating {model_name}...")
    accuracy, report, avg_loss = evaluate_model(model, test_loader, criterion)
    print(f"{model_name} - Loss: {avg_loss:.4f}")
    print(f"{model_name} - Accuracy: {accuracy:.4f}")

    # Lưu mô hình
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"{model_name} saved to: {model_path}")

    # Lưu kết quả
    save_results(model_name, accuracy, report, RESULTS_DIR)

# Khởi tạo và huấn luyện các mô hình
input_size = X_train.shape[1]
num_classes = len(set(y_train))

# Huấn luyện và đánh giá MLP
mlp_model = MLPClassifier(input_size, output_dim=num_classes)
train_and_evaluate(mlp_model, "MLPClassifier", train_loader, test_loader)

# Huấn luyện và đánh giá CNN
cnn_model = CNNClassifier(input_size, output_dim=num_classes)
train_and_evaluate(cnn_model, "CNNClassifier", train_loader, test_loader)

# Huấn luyện và đánh giá RNN
rnn_model = RNNClassifier(input_size, output_dim=num_classes)
train_and_evaluate(rnn_model, "RNNClassifier", train_loader, test_loader)