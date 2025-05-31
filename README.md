## Project Title
20 Newsgroups Text Classification: A Comparative Study of Traditional ML and Deep Learning Models

## Overview
Dự án này tập trung vào việc phân loại văn bản từ tập dữ liệu 20 Newsgroups, một bộ dữ liệu tiêu chuẩn trong lĩnh vực phân loại văn bản. Mục tiêu chính là khám phá và so sánh hiệu quả của các phương pháp học máy đa dạng, từ các thuật toán truyền thống như Naïve Bayes, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest đến các kiến trúc học sâu tiên tiến bao gồm Multi-Layer Perceptron (MLP), Recurrent Neural Networks (RNN), và đặc biệt là mô hình dựa trên Transformer như BERT.

Các nguyên tắc và phương pháp được trình bày trong dự án này có thể được áp dụng rộng rãi cho nhiều bài toán phân loại văn bản thực tế, bao gồm nhưng không giới hạn ở: phân loại chủ đề tin tức, phân tích cảm xúc, phát hiện thư rác (spam), kiểm duyệt nội dung, định tuyến tài liệu và hệ thống gợi ý dựa trên nội dung văn bản.

## Project Structure
Dự án được tổ chức một cách rõ ràng và khoa học để dễ dàng quản lý, phát triển và tái tạo kết quả. Dưới đây là cấu trúc thư mục chi tiết:

project_name/
│
├── configs/
│
├── data/
│ ├── raw/
│ ├── interim/
│ └── processed/
│
├── models/
│ ├── baseline_model.pkl
│ └── deep_learning_model.pth
│
├── notebooks/
│ ├── 1_EDA.ipynb
│ ├── 2_Baseline_Models.ipynb
│ └── 3_DeepLearning.ipynb
│
├── results/
│ ├── figures/
│ ├── logs/
│ └── tables/
│
├── src/
│ ├── pycache/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── models/
│ │ ├── baseline.py
│ │ └── deep_models.py
│ ├── evaluation.py
│ └── utils.py
│
├── requirements.txt
└── README.md


## Installation

### Yêu cầu hệ thống
*   Python 3.8 hoặc cao hơn.
*   Các thư viện Python được liệt kê trong `requirements.txt`.
*   Khuyến nghị sử dụng môi trường ảo (virtual environment) để quản lý dependency.

### Cài đặt thư viện
Để cài đặt tất cả các thư viện cần thiết, hãy chạy lệnh sau từ thư mục gốc của dự án:
pip install -r requirements.txt

Cài đặt dữ liệu
Tập dữ liệu 20 Newsgroups có thể được tải về tự động bởi thư viện scikit-learn khi bạn chạy các notebook hoặc script xử lý dữ liệu. Nếu bạn muốn sử dụng một phiên bản cụ thể hoặc dữ liệu đã được tùy chỉnh, bạn có thể tải xuống từ Kaggle và đặt nó vào thư mục data/raw/.


### Chạy code
Chạy code nguồn
1. EDA.ipynb: Khám phá dữ liệu ban đầu.
2. python src/data_preprocessing.py: Tiền xử lý dữ liệu
3. python src/models/baseline.py: Huấn luyện mô hình Baseline
4. Huấn luyện mô hình Deep Learning: Chạy các jupiter notebook: 
    src/models/deep_models_TF-IDF.ipynb: Huấn luyện mô hình MLP với kỹ thuật TI-IDF
    src/models/deep_models_glove.ipynb: Huấn luyện thử nghiệm các mô hình RNN, CNN, Transformer cho classification
5. Đánh giá và so sánh các mô hình: Chạy các jupiter notebook:
    notebooks/2_Baseline_Models.ipynb
    notebooks/3_Deep_Learning.ipynb

### Kết quả thu được
Kết quả thu được
1. Các metric kết quả phân loại của các mô hình
    Dự án sử dụng các chỉ số đánh giá hiệu suất mô hình phổ biến trong phân loại văn bản để cung cấp cái nhìn toàn diện về hiệu suất của từng mô hình:
        Baseline: results/tables/baseline_results.csv
        Deep models: results/tables/deep_models_results.csv

2. Visualization
    Sử dụng biểu đồ cột để so shán các kết quả đạt được, các biểu đồ so sánh được lưu tại: 
        results/figures