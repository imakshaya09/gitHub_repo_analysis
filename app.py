import json
import os
import pickle
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = BASE_DIR / "models"
PROCESSED_PATH = DATA_DIR / "processed.json"
MODEL_PATH = MODEL_DIR / "classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

DATASET_ENV_VAR = "GITHUB_DATA_CSV"
CATEGORIES = ["kaggle", "study", "professional"]

app = Flask(__name__)

CATEGORY_KEYWORDS = {
    "kaggle": ["kaggle", "kagglers", "kaggle.com", "kaggle notebook", "kaggle dataset"],
    "study": ["tutorial", "learning", "learn", "study", "practice", "course", "assignment", "problem", "exercise", "notebook"],
    "professional": ["production", "professional", "enterprise", "business", "service", "library", "framework", "api", "tool", "deployment", "system"]
}

MANUAL_TRAINING_DATA = [
    ("This repository contains Kaggle notebooks and dataset exploration for competition practice", "kaggle"),
    ("A Kaggle kernel used to analyze Titanic dataset and prepare submission", "kaggle"),
    ("Python exercises for students learning data science and machine learning", "study"),
    ("Tutorial code for building an image classifier from scratch", "study"),
    ("Course materials for an introductory programming class", "study"),
    ("A professional web service for validating GitHub repository metadata", "professional"),
    ("Enterprise-grade deployment tooling for microservices", "professional"),
    ("Production-ready REST API written in Flask", "professional"),
]

COLUMN_ALIASES = {
    "language": ["language", "repo_language", "primary_language", "primary language", "lang"],
    "description": ["description", "repo_description", "about", "summary"],
    "stars": ["stars", "stars count", "stargazers_count", "stargazers count", "watchers", "watchers_count", "star_count", "repo_stars"],
    "url": [
        "github_url", "url", "repo_url", "html_url", "repository_url",
        "full_name", "full name", "repo_name", "repository_name", "name"
    ]
}


def ensure_directories():
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


def find_csv_file():
    env_path = os.environ.get(DATASET_ENV_VAR)
    if env_path:
        csv_path = Path(env_path).expanduser()
        if csv_path.exists() and csv_path.is_file():
            return csv_path
        raise FileNotFoundError(f"CSV file not found at {DATASET_ENV_VAR}={env_path}")

    data_csv = DATA_DIR / "github_top_repositories.csv"
    if data_csv.exists():
        return data_csv

    local_csv = BASE_DIR / "github_top_repositories.csv"
    if local_csv.exists():
        return local_csv

    default_csv = Path.home() / "Documents/2026_job_search/kaggle_dataset/github_top_repositories.csv"
    if default_csv.exists():
        return default_csv

    raise FileNotFoundError(
        "No CSV file found. Place the file in data/github_top_repositories.csv or set the environment variable "
        f"{DATASET_ENV_VAR} to the CSV path."
    )


def normalize_column(df, names):
    if isinstance(names, str):
        names = [names]

    for name in names:
        candidates = COLUMN_ALIASES.get(name, [name])
        for candidate in candidates:
            matches = [col for col in df.columns if col.lower() == candidate.lower()]
            if matches:
                return matches[0]
    return None


def infer_columns(df):
    language_col = normalize_column(df, "language")
    description_col = normalize_column(df, "description")
    stars_col = normalize_column(df, "stars")
    url_col = normalize_column(df, "url")

    if not description_col:
        raise ValueError("Description column not found in dataset.")
    if not language_col:
        raise ValueError("Language column not found in dataset.")
    if not stars_col:
        raise ValueError("Stars column not found in dataset.")
    if not url_col:
        raise ValueError("URL column not found in dataset.")

    return language_col, description_col, stars_col, url_col


def parse_csv_file():
    csv_path = find_csv_file()
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    language_col, description_col, stars_col, url_col = infer_columns(df)
    df = df[[language_col, description_col, stars_col, url_col]].copy()
    df.columns = ["language", "description", "stars", "url"]

    df["language"] = df["language"].fillna("Unknown").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0).astype(int)

    df["url"] = df["url"].apply(normalize_repo_url)
    df = df[df["url"] != ""].reset_index(drop=True)

    return df


def normalize_repo_url(url):
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if "/" in url and not url.startswith("github.com"):
        return f"https://github.com/{url.lstrip('/')}"
    if url.startswith("github.com"):
        return f"https://{url}"
    return url


def bootstrap_training_examples(df):
    samples = []
    for _, row in df.iterrows():
        text = str(row["description"]).lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                samples.append((row["description"], category))
                break
        if len(samples) >= 200:
            break

    samples.extend(MANUAL_TRAINING_DATA)
    unique_samples = {}
    for text, label in samples:
        key = (text.strip().lower(), label)
        unique_samples[key] = (text, label)

    training_data = list(unique_samples.values())
    return training_data


def build_classifier(df):
    training_data = bootstrap_training_examples(df)
    if len(training_data) < 10:
        raise ValueError("Not enough labeled training examples to build a classifier.")

    texts, labels = zip(*training_data)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=2000)),
        ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear"))),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def load_or_train_classifier(df):
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                classifier = pickle.load(f)
            return classifier
        except Exception:
            pass

    classifier = build_classifier(df)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(classifier, f)
    return classifier


def classify_description(classifier, description):
    if not description:
        return "study"
    try:
        category = classifier.predict([description])[0]
        if category not in CATEGORIES:
            return "study"
        return category
    except (NotFittedError, ValueError):
        return heuristics_category(description)


def heuristics_category(description):
    text = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "professional"


def group_repositories(df, classifier):
    repos = []
    for _, row in df.iterrows():
        category = classify_description(classifier, row["description"])
        repos.append({
            "language": row["language"] or "Unknown",
            "github_url": row["url"],
            "stars": int(row["stars"]),
            "description": row["description"],
            "category": category,
        })

    groups = {category: [] for category in CATEGORIES}
    for repo in repos:
        groups[repo["category"]].append(repo)

    for category in CATEGORIES:
        groups[category] = sorted(groups[category], key=lambda item: item["stars"], reverse=True)
    return groups


def build_language_stats(groups):
    stats = {}
    for category, repos in groups.items():
        counts = {}
        total = max(len(repos), 1)
        for repo in repos:
            language = repo["language"] or "Unknown"
            counts[language] = counts.get(language, 0) + 1
        stats[category] = [
            {"language": language, "count": count, "percent": round(count / total * 100, 1)}
            for language, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]
    return stats


def save_processed_data(data):
    with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_processed_data():
    if not PROCESSED_PATH.exists():
        return None
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def refresh_processed_data(force_download=False):
    ensure_directories()

    df = parse_csv_file()
    classifier = load_or_train_classifier(df)
    groups = group_repositories(df, classifier)
    stats = build_language_stats(groups)

    result = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "categories": groups,
        "language_stats": stats,
    }
    save_processed_data(result)
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    data = load_processed_data()
    if data is None:
        try:
            data = refresh_processed_data(force_download=False)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    return jsonify(data)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        force_download = request.args.get("force", "false").lower() == "true"
        data = refresh_processed_data(force_download=force_download)
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    ensure_directories()
    app.run(debug=True)
