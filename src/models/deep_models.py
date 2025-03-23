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

# Kiểm tra xem có sự chồng chéo giữa các tập dữ liệu không
if 'id' in train_df.columns:
    train_ids = set(train_df['id'])
    val_ids = set(val_df['id'])
    test_ids = set(test_df['id'])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Train và validation có sự chồng chéo"
    assert len(train_ids.intersection(test_ids)) == 0, "Train và test có sự chồng chéo"
    assert len(val_ids.intersection(test_ids)) == 0, "Validation và test có sự chồng chéo"

# Vector hóa văn bản bằng TF-IDF (dùng tập train để fit vectorizer)
X_train, vectorizer = tfidf_vectorize_ngrams(train_df['clean_text'])
y_train = train_df['label']

# Kiểm tra xem y_train có chứa giá trị NaN không
if y_train.isna().any():
    print(f"Cảnh báo: Tìm thấy {y_train.isna().sum()} giá trị NaN trong nhãn tập train")
    # Xử lý theo cách phù hợp với bài toán, ví dụ: loại bỏ hoặc gán nhãn mặc định
    # KHÔNG sử dụng fillna(0) vì có thể gây nhầm lẫn với nhãn 0
    # Dưới đây là cách loại bỏ các hàng có nhãn NaN
    valid_indices = ~y_train.isna()
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]

X_val = vectorizer.transform(val_df['clean_text'])
y_val = val_df['label']
if y_val.isna().any():
    valid_indices = ~y_val.isna()
    X_val = X_val[valid_indices]
    y_val = y_val[valid_indices]

X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['label']
if y_test.isna().any():
    valid_indices = ~y_test.isna()
    X_test = X_test[valid_indices]
    y_test = y_test[valid_indices]

# Kiểm tra xem các nhãn đã được chuyển đổi thành số chưa
if not np.issubdtype(y_train.dtype, np.number):
    # Chuyển đổi nhãn thành số nếu chưa phải
    # Nên sử dụng một encoder để đảm bảo nhất quán
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)
    y_test = encoder.transform(y_test)
    
    # Lưu encoder để sử dụng sau này
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"Đã chuyển đổi nhãn từ {encoder.classes_} thành {list(range(len(encoder.classes_)))}")

# Chuyển đổi dữ liệu thành tensor
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(set(y_train))
print(f"Số lớp nhãn: {num_classes}")

# Mô hình MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.5, output_dim=num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Thêm dropout để giảm overfitting
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Áp dụng dropout
        x = self.fc2(x)
        return x

# Mô hình CNN được điều chỉnh cho dữ liệu TF-IDF
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=num_classes, num_filters=100, dropout_rate=0.5):
        super(CNNClassifier, self).__init__()
        # Sử dụng MLP với nhiều lớp hidden thay vì CNN
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        # Không sử dụng unsqueeze vì dữ liệu TF-IDF không có cấu trúc không gian
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Mô hình RNN được điều chỉnh cho dữ liệu TF-IDF
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=num_classes, dropout_rate=0.5):
        super(RNNClassifier, self).__init__()
        # Sử dụng MLP nhiều lớp thay vì LSTM
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Không sử dụng unsqueeze vì dữ liệu TF-IDF không có cấu trúc tuần tự
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Hàm huấn luyện với early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3):
    model.train()
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # Huấn luyện
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Đánh giá trên tập validation
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(y_batch.cpu().numpy())
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Kiểm tra early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            print(f"Lưu mô hình tốt nhất với Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Không cải thiện: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping sau {epoch+1} epochs")
                break
    
    # Khôi phục mô hình tốt nhất
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# Hàm đánh giá
def evaluate_model(model, data_loader, criterion):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    # zero_division=0 để tránh lỗi khi một lớp không có mẫu nào
    report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    avg_loss = total_loss / len(data_loader)
    
    # In confusion matrix để phân tích chi tiết
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    return accuracy, report, avg_loss

# Hàm lưu kết quả
def save_results(model_name, accuracy, report, val_accuracy, val_report, results_dir):
    results = {
        "Model": model_name,
        "Test_Accuracy": accuracy,
        "Test_Precision": report['weighted avg']['precision'],
        "Test_Recall": report['weighted avg']['recall'],
        "Test_F1-Score": report['weighted avg']['f1-score'],
        "Val_Accuracy": val_accuracy,
        "Val_Precision": val_report['weighted avg']['precision'],
        "Val_Recall": val_report['weighted avg']['recall'],
        "Val_F1-Score": val_report['weighted avg']['f1-score']
    }
    results_df = pd.DataFrame([results])
    results_file = os.path.join(results_dir, "deep_models_results.csv")

    # Check if the file exists. If not, create a new file with headers.
    if not os.path.exists(results_file):
        results_df.to_csv(results_file, index=False, header=True)
    else:
        results_df.to_csv(results_file, index=False, header=False, mode='a')  # Append without headers

    print(f"{model_name} results saved to: {results_file}")

# Hàm chính để huấn luyện và đánh giá mô hình
def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, num_epochs=0, learning_rate=0.001, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Thêm L2 regularization

    print(f"Training {model_name}...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=num_epochs, patience=patience)

    print(f"Evaluating {model_name} on validation set...")
    val_accuracy, val_report, val_avg_loss = evaluate_model(model, val_loader, criterion)
    print(f"{model_name} - Val Loss: {val_avg_loss:.4f}")
    print(f"{model_name} - Val Accuracy: {val_accuracy:.4f}")
    
    print(f"Evaluating {model_name} on test set...")
    test_accuracy, test_report, test_avg_loss = evaluate_model(model, test_loader, criterion)
    print(f"{model_name} - Test Loss: {test_avg_loss:.4f}")
    print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}")

    # Lưu mô hình
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"{model_name} saved to: {model_path}")

    # Lưu kết quả
    save_results(model_name, test_accuracy, test_report, val_accuracy, val_report, RESULTS_DIR)

# Khởi tạo và huấn luyện các mô hình
input_size = X_train.shape[1]
num_classes = len(set(y_train))

print(f"Kích thước đầu vào (số features TF-IDF): {input_size}")
print(f"Số lượng lớp: {num_classes}")
print(f"Kích thước tập train: {len(train_dataset)}")
print(f"Kích thước tập validation: {len(val_dataset)}")
print(f"Kích thước tập test: {len(test_dataset)}")

# Huấn luyện và đánh giá MLP
mlp_model = MLPClassifier(input_size, hidden_dim=128, dropout_rate=0.5, output_dim=num_classes)
train_and_evaluate(mlp_model, "MLPClassifier", train_loader, val_loader, test_loader, num_epochs=15, patience=3)

# Huấn luyện và đánh giá "CNN" (thực chất là MLP nhiều lớp)
cnn_model = CNNClassifier(input_size, output_dim=num_classes, dropout_rate=0.5)
train_and_evaluate(cnn_model, "CNNClassifier", train_loader, val_loader, test_loader, num_epochs=15, patience=3)

# Huấn luyện và đánh giá "RNN" (thực chất là MLP nhiều lớp)
rnn_model = RNNClassifier(input_size, output_dim=num_classes, dropout_rate=0.5)
train_and_evaluate(rnn_model, "RNNClassifier", train_loader, val_loader, test_loader, num_epochs=15, patience=3)