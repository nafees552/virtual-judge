"""Train interpretable ML models using real datasets for toxicity, sarcasm, and sentiment.

Adheres to FAST NUCES AI Project Proposal:
- Layer 1: Logistic Regression text classifiers with TF-IDF features.
- Offline persistence into ml_models/ (.pkl/.joblib).
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


RANDOM_STATE = 42


def build_pipeline():
    """Returns the standardized scikit-learn pipeline configuration."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def load_toxicity_dataset(csv_path, target_samples=6000):
    """Loads and balances Davidson/Jigsaw hate & offensive language data into binary toxicity."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Toxicity dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    text_col = "tweet" if "tweet" in df.columns else "text"

    if "class" in df.columns:
        # class: 0=hate speech, 1=offensive language, 2=neither
        non_toxic = df[df["class"] == 2].copy()
        non_toxic["label"] = 0
        toxic = df[df["class"].isin([0, 1])].copy()
        toxic["label"] = 1

        n_per_class = min(len(non_toxic), len(toxic), target_samples // 2)
        balanced = pd.concat([
            non_toxic.sample(n=n_per_class, random_state=RANDOM_STATE),
            toxic.sample(n=n_per_class, random_state=RANDOM_STATE),
        ]).sample(frac=1.0, random_state=RANDOM_STATE)

        texts = balanced[text_col].astype(str).tolist()
        labels = balanced["label"].astype(int).tolist()
        return texts, labels
    else:
        df["label"] = df["label"].astype(int)
        df = df.dropna(subset=[text_col, "label"])
        return df[text_col].astype(str).tolist(), df["label"].tolist()


def load_sarcasm_dataset(json_paths, max_samples=10000):
    """Loads sarcasm headlines dataset from JSONL files."""
    rows = []
    for path in json_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "headline" in row and "is_sarcastic" in row:
                    rows.append((str(row["headline"]), int(row["is_sarcastic"])))

    if not rows:
        raise FileNotFoundError("Sarcasm dataset not found.")

    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates(subset=["text"])
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_STATE)

    return df["text"].tolist(), df["label"].tolist()


def load_sentiment_dataset(tsv_path):
    """Loads SST-2 sentiment dataset (tab-separated: sentence, label)."""
    if not tsv_path.exists():
        raise FileNotFoundError(f"Sentiment dataset not found at {tsv_path}")

    texts = []
    labels = []
    with tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                texts.append(parts[0])
                # Convert 0/1 to negative/positive labels
                labels.append("positive" if parts[1].strip() == "1" else "negative")

    return texts, labels


def train_and_save(name, texts, labels, output_path):
    """Fits model, displays evaluation metrics, and persists via joblib."""
    print(f"\n--- Training {name.upper()} Model ({len(texts)} samples) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"=== {name.upper()} CLASSIFICATION REPORT ===")
    print(classification_report(y_test, preds))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path, protocol=4)
    print(f"Saved model to: {output_path}")
    return model


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    model_dir = base_dir / "ml_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Toxicity
    tox_csv = data_dir / "labeled_data.csv"
    tox_texts, tox_labels = load_toxicity_dataset(tox_csv)
    train_and_save("toxicity", tox_texts, tox_labels, model_dir / "toxicity_model.pkl")

    # 2. Sarcasm
    sar_paths = [
        data_dir / "Sarcasm_Headlines_Dataset.json",
        data_dir / "Sarcasm_Headlines_Dataset_v2.json",
    ]
    sar_texts, sar_labels = load_sarcasm_dataset(sar_paths)
    train_and_save("sarcasm", sar_texts, sar_labels, model_dir / "sarcasm_model.pkl")

    # 3. Sentiment
    sen_tsv = data_dir / "sst2_sentiment.tsv"
    sen_texts, sen_labels = load_sentiment_dataset(sen_tsv)
    train_and_save("sentiment", sen_texts, sen_labels, model_dir / "sentiment_model.pkl")

    print("\nAll 3 Logistic Regression models successfully trained and serialized!")


if __name__ == "__main__":
    main()
