# GitHub Repository Analyzer

A simple Flask application that downloads a Kaggle GitHub repositories dataset, classifies repository descriptions into `kaggle`, `study`, or `professional`, and displays the results in a tabbed UI.

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Place the provided CSV file in `data/github_top_repositories.csv`, or set the `GITHUB_DATA_CSV` environment variable:

```bash
export GITHUB_DATA_CSV=/path/to/github_top_repositories.csv
```

3. Run the app:

```bash
python app.py
```

4. Open `http://127.0.0.1:5000` in your browser.

## Project structure

- `app.py` — Flask app, Kaggle download, classification, caching
- `templates/index.html` — UI with tabs and charts
- `data/` — raw dataset and processed cache
- `models/` — trained classifier model cache

## Notes

- The first request may take longer because the Kaggle dataset is downloaded and the classifier is trained.
- Use the refresh button in the UI to reload data and recalculate categories.
