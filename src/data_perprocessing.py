# File: src/data_preprocessing.py

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

def load_raw_data(root_dir):
    """
    Duyệt qua thư mục raw data, đọc các file văn bản và gán nhãn dựa trên tên thư mục.
    
    Args:
        root_dir (str): Đường dẫn tới thư mục chứa các thư mục con của các nhãn.
    
    Returns:
        DataFrame: Chứa hai cột 'text' và 'label'.
    """
    categories = os.listdir(root_dir)

    rows = []
    for cat in categories:
        cat_path = os.path.join(root_dir, cat)
        if os.path.isdir(cat_path):
            for filename in os.listdir(cat_path):
                file_path = os.path.join(cat_path, filename)
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
                rows.append((text, cat))

    df = pd.DataFrame(rows, columns=['text', 'label'])
    return df

def text_preprocess(text, stop_words, lemmatizer):
    """
    Tiền xử lý văn bản: chuyển về chữ thường, loại bỏ HTML, tokenization, loại bỏ stopwords và lemmatization.
    
    Args:
        text (str): Văn bản đầu vào.
        stop_words (set): Tập từ dừng.
        lemmatizer (WordNetLemmatizer): Đối tượng lemmatizer.
        
    Returns:
        str: Văn bản đã được xử lý.
    """
    # Chuyển thành chữ thường
    text = text.lower()
    # Loại bỏ HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Tách từ bằng regex (chỉ lấy các ký tự chữ và số, ở đây lấy cả chữ cái)
    tokens = re.findall(r'\w+', text)
    # Lọc: chỉ giữ những token chứa chữ cái và có độ dài > 1
    tokens = [token for token in tokens if token.isalpha() and len(token) > 1]
    # Loại bỏ stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Thực hiện lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def preprocess_data(df):
    """
    Áp dụng tiền xử lý cho cột 'text' trong DataFrame.
    
    Args:
        df (DataFrame): Dữ liệu thô.
    
    Returns:
        DataFrame: Dữ liệu sau khi xử lý, thêm cột 'clean_text'.
    """
    # Tải stopwords và wordnet (nếu chưa có)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    df['clean_text'] = df['text'].apply(lambda x: text_preprocess(x, stop_words, lemmatizer))
    return df

def main():
    # Đường dẫn đến dữ liệu
    root_dir = './data/20_newsgroups'
    processed_file = './data/processed/20news_processed.csv'
    
    train_file = './data/processed/train.csv'
    val_file = './data/processed/val.csv'
    test_file = './data/processed/test.csv'
    
    # Load & xử lý dữ liệu
    print("Loading raw data...")
    df = load_raw_data(root_dir)
    print(f"Raw data loaded. Shape: {df.shape}")

    print("Starting text preprocessing...")
    df = preprocess_data(df)
    print("Preprocessing complete. Sample processed text:")
    print(df[['clean_text', 'label']].head())

    # Tạo thư mục processed nếu chưa có
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)

    # Lưu toàn bộ dữ liệu vào 20news_processed.csv
    df.to_csv(processed_file, index=False)
    print(f"Full processed data saved to: {processed_file}")

    # Chia dữ liệu thành train, val, test (80-10-10)
    train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

    # Lưu các tệp train, val, test
    train.to_csv(train_file, index=False)
    val.to_csv(val_file, index=False)
    test.to_csv(test_file, index=False)

    print(f"Train data saved to: {train_file} (Size: {train.shape})")
    print(f"Validation data saved to: {val_file} (Size: {val.shape})")
    print(f"Test data saved to: {test_file} (Size: {test.shape})")

if __name__ == '__main__':
    main()

# import os
# data_path = "data/20_newsgroups"
# if os.path.exists(data_path):
#     print(f"Directory '{data_path}' exists.")
# else:
#     print(f"Directory '{data_path}' does NOT exist.")
