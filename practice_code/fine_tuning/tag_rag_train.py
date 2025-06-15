# tag_rag_train.py using scikit-learn
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import joblib

# Load Q&A JSON
with open("qa_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract user-assistant pairs
inputs, outputs = [], []
for item in data:
    user_msg = next(msg["content"] for msg in item["messages"] if msg["role"] == "user")
    assistant_msg = next(msg["content"] for msg in item["messages"] if msg["role"] == "assistant")
    inputs.append(user_msg)
    outputs.append(assistant_msg)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(inputs)

# KNN 모델 (유사도 기반)
knn = NearestNeighbors(n_neighbors=1, metric='cosine')
knn.fit(X)

# Save
joblib.dump(knn, "qa_knn_model.joblib")
joblib.dump(vectorizer, "qa_tfidf_vectorizer.joblib")

# Save outputs
with open("qa_responses.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)
