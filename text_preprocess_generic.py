# -*- coding: utf-8 -*-
"""
Generic text preprocessing + TF-IDF + optional noise injection
-------------------------------------------------------------
- Removes any hard-coded dataset names and paths.
- Accepts one or more source glob patterns via CLI, each paired with a string label.
- Converts labels to numeric using a provided mapping.
- Optionally injects synthetic noise samples drawn from the learned vocabulary.

Example usage:
    python text_preprocess_generic.py \
      --src "./data/neg_truth/*/,truthful" \
      --src "./data/neg_deceptive/*/,deceptive" \
      --label-map "truthful=1" "deceptive=-1" \
      --max-features 1000 \
      --noise-rate 0.3 \
      --seed 42 \
      --save-prefix out/yelp_neg

Outputs when --save-prefix is set:
    out/yelp_neg_X.npy                (TF-IDF dense features for clean data)
    out/yelp_neg_y.csv                (labels for clean data; last column is y)
    out/yelp_neg_X_noise.npy          (TF-IDF dense features for noise data; if noise-rate>0)
    out/yelp_neg_y_noise.csv          (labels for noise data; if noise-rate>0)
    out/yelp_neg_X_combined.npy       (clean + noise)
    out/yelp_neg_y_combined.csv

If you don't pass --save-prefix, the script just prints dataset shapes.
"""
import argparse
import re
from glob import glob
import random
import string
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Stop words (customizable)
# -------------------------
DEFAULT_STOP_WORDS = [
    "i","me","my","myself","we","our","ours","ourselves","he","him","his","himself","she",
    "her","hers","herself","you","your","yours","yourself","yourselves","it","its","itself",
    "they","them","their","theirs","themselves","have","has","had","having","do","does","did",
    "doing","what","which","who","whom","this","that","these","those","am","is","are","was",
    "were","be","been","being","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","then","once","here","there","when","where","why",
    "how","all","about","against","between","into","through","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","during","before","after",
    "above","below","to","from","up","down","in","out","on","off","over","under","again",
    "further","any","both","each","few","more","most","other","some","such","no","nor","not",
    "didnt","dont","doesnt","isnt","arent","wasnt","werent","havent","hasnt","hadnt","shouldnt"
]

# -------------------------
# Text utilities
# -------------------------

def preprocess_text(text: str, stop_words: List[str]) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\d+", "", text)        # remove digits
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


def read_glob_preprocess(glob_pattern: str, label: str, stop_words: List[str]) -> Tuple[List[str], List[str]]:
    files = glob(glob_pattern)
    texts, labels = [], []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
            texts.append(preprocess_text(raw, stop_words))
            labels.append(label)
    return texts, labels

# -------------------------
# Noise generation
# -------------------------

def random_string(length: int = 20, vocab: np.ndarray = None) -> str:
    if vocab is not None and len(vocab) > 0:
        # sample words from vocab to make plausible-ish noise
        k = min(length, len(vocab))
        words = random.sample(list(vocab), k)
        return " ".join(words)
    # fallback: alphanumeric noise
    letters = string.ascii_letters + string.digits + " "
    return "".join(random.choice(letters) for _ in range(length))


def generate_random_comments(n: int, vocab: np.ndarray) -> Tuple[List[str], List[int]]:
    comments, labels = [], []
    for _ in range(n):
        comment = random_string(random.randint(10, 50), vocab=vocab)
        label = random.choice([1, -1])
        comments.append(comment)
        labels.append(label)
    return comments, labels

# -------------------------
# Main
# -------------------------

def parse_src_args(src_args: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for s in src_args:
        if "," not in s:
            raise ValueError(f"--src expects 'glob,label' pairs, got: {s}")
        glob_pat, label = s.split(",", 1)
        pairs.append((glob_pat, label))
    return pairs


def parse_label_map(pairs: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--label-map expects 'name=int' entries, got: {p}")
        k, v = p.split("=", 1)
        mapping[k] = int(v)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Generic text preprocessing with TF-IDF and optional noise")
    parser.add_argument("--src", action="append", required=True,
                        help="Add sources as 'glob,label' (can be repeated)")
    parser.add_argument("--label-map", nargs="*", default=[],
                        help="Label mapping as 'name=int' (e.g., truthful=1 deceptive=-1)")
    parser.add_argument("--max-features", type=int, default=1000, help="TF-IDF max features")
    parser.add_argument("--noise-rate", type=float, default=0.0, help="Fraction of noise relative to clean samples (0..1)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--save-prefix", type=str, default=None, help="If set, save arrays/labels with this prefix")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    stop_words = DEFAULT_STOP_WORDS

    # Load and preprocess
    texts, labels = [], []
    for glob_pat, label in parse_src_args(args.src):
        t, l = read_glob_preprocess(glob_pat, label, stop_words)
        texts.extend(t); labels.extend(l)

    assert len(texts) == len(labels), "#texts must equal #labels"

    df = pd.DataFrame({"text": texts, "label": labels})

    # Map string labels -> ints if mapping provided
    if args.label_map:
        name2id = parse_label_map(args.label_map)
        df["label"] = df["label"].map(lambda x: name2id.get(x, x))
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X = vectorizer.fit_transform(df["text"]).toarray()
    y = np.array(df["label"], dtype=object)

    vocab = vectorizer.get_feature_names_out()

    # Optional noise
    X_noise = None; y_noise = None
    if args.noise_rate and args.noise_rate > 0:
        n_noise = int(args.noise_rate * len(y))
        noise_texts, noise_labels = generate_random_comments(n_noise, vocab)
        X_noise = vectorizer.transform(noise_texts).toarray()
        y_noise = np.array(noise_labels)
        X_combined = np.vstack([X, X_noise])
        y_combined = np.hstack([y, y_noise])
    else:
        X_combined, y_combined = X, y

    # Report
    print("Clean X shape:", X.shape)
    print("Clean y shape:", y.shape)
    if X_noise is not None:
        print("Noise X shape:", X_noise.shape)
        print("Noise y shape:", y_noise.shape)
    print("Combined X shape:", X_combined.shape)
    print("Combined y shape:", y_combined.shape)

    # Optional save
    if args.save_prefix:
        # clean
        np.save(f"{args.save_prefix}_X.npy", X)
        pd.DataFrame({"y": y}).to_csv(f"{args.save_prefix}_y.csv", index=False)
        # noise
        if X_noise is not None:
            np.save(f"{args.save_prefix}_X_noise.npy", X_noise)
            pd.DataFrame({"y": y_noise}).to_csv(f"{args.save_prefix}_y_noise.csv", index=False)
        # combined
        np.save(f"{args.save_prefix}_X_combined.npy", X_combined)
        pd.DataFrame({"y": y_combined}).to_csv(f"{args.save_prefix}_y_combined.csv", index=False)
        # vocab
        pd.DataFrame({"vocab": vocab}).to_csv(f"{args.save_prefix}_vocab.csv", index=False)
        print(f"Saved arrays and labels with prefix: {args.save_prefix}")

if __name__ == "__main__":
    main()
