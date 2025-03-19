import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def tfidf_vectorize(texts, max_features=5000):
    """Chuyển đổi văn bản thành vector TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def count_vectorize(texts, max_features=5000):
    """Chuyển đổi văn bản thành vector dựa trên số lần xuất hiện (BoW)."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def tfidf_vectorize_ngrams(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))  # Thêm n-grams
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def count_vectorize_ngrams(texts, max_features=5000):
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))  # Thêm n-grams
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

if __name__ == "__main__":
    # Test nhanh với dữ liệu giả lập
    sample_texts = ["This is a sample text", "Machine learning is amazing", "Natural language processing with Python"]
    tfidf_features, tfidf_vectorizer = tfidf_vectorize(sample_texts)
    print("TF-IDF shape:", tfidf_features.shape)
