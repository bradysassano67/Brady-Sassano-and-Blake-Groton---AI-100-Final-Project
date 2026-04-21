import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

WELFAKE_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)          # remove urls
    s = re.sub(r"[^a-z0-9\s]", " ", s)               # remove punctuation/symbols
    s = re.sub(r"\s+", " ", s).strip()               # collapse spaces
    return s


def plot_history(history, out_dir: str):
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.title("Training vs Validation Accuracy")
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, out_dir: str):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Fake(0)", "Real(1)"])
    plt.yticks([0, 1], ["Fake(0)", "Real(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("Loading dataset:", WELFAKE_PATH)
    df = pd.read_csv(WELFAKE_PATH)

    # WELFake columns: ['Unnamed: 0', 'title', 'text', 'label']
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])

    # Combine title + text for a stronger signal
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["combined"] = (df["title"] + " " + df["text"]).map(clean_text)

    df = df[df["combined"].str.len() > 10].copy()
    df = df[df["label"].isin([0, 1])].copy()

    print("Rows after cleaning:", df.shape[0])
    print("Label counts:\n", df["label"].value_counts())

    X = df["combined"].values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize
    vocab_size = 30000
    max_len = 400

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(test_seq, maxlen=max_len, padding="post", truncating="post")

    # Model
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=6,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
    print("\nTest Accuracy:", round(float(test_acc), 4))
    print("Test Loss:", round(float(test_loss), 4))

    probs = model.predict(X_test_pad, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, preds, digits=4))

    # Save outputs
    plot_history(history, OUT_DIR)
    plot_confusion(cm, OUT_DIR)

    model_path = os.path.join(OUT_DIR, "fake_news_lstm.keras")
    model.save(model_path)

    tok_path = os.path.join(OUT_DIR, "tokenizer.json")
    with open(tok_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    meta = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss)
    }
    with open(os.path.join(OUT_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print("-", model_path)
    print("-", tok_path)
    print("-", os.path.join(OUT_DIR, "accuracy_curve.png"))
    print("-", os.path.join(OUT_DIR, "loss_curve.png"))
    print("-", os.path.join(OUT_DIR, "confusion_matrix.png"))
    print("-", os.path.join(OUT_DIR, "run_metadata.json"))


if __name__ == "__main__":
    main()