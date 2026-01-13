import os
import json

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# PATH CONFIGURATION
# -------------------------
TRAIN_PATH = "train"
TEST_PATH = "test"

# -------------------------
# DATA LOADING
# -------------------------
def collect_samples(root_dir):
    texts = []
    labels = []

    for topic in os.listdir(root_dir):
        topic_path = os.path.join(root_dir, topic)

        if not os.path.isdir(topic_path):
            continue

        for file_name in tqdm(os.listdir(topic_path), desc=f"Reading {topic}"):
            if file_name.endswith(".json"):
                file_path = os.path.join(topic_path, file_name)

                with open(file_path, "r", encoding="utf-8") as file:
                    record = json.load(file)

                texts.append(record["problem"])
                labels.append(topic)

    return texts, labels


train_texts, train_labels = collect_samples(TRAIN_PATH)
test_texts, test_labels = collect_samples(TEST_PATH)

print(f"Training samples: {len(train_texts)}")
print(f"Testing samples: {len(test_texts)}")

# -------------------------
# EXPERIMENTS
# -------------------------
vector_spaces = {
    "tfidf": TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=6000
    ),
    "bow": CountVectorizer(
        lowercase=True,
        max_features=6000
    )
}

for name, vectorizer in vector_spaces.items():
    print(f"\n--- Vectorizer: {name.upper()} ---")

    train_vectors = vectorizer.fit_transform(train_texts)
    test_vectors = vectorizer.transform(test_texts)

    classifier = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    classifier.fit(train_vectors, train_labels)
    predictions = classifier.predict(test_vectors)

    acc = accuracy_score(test_labels, predictions)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(test_labels, predictions))
