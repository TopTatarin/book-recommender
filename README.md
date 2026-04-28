# Semantic Book Recommender

A desktop application that recommends books based on a natural-language description, category, and emotional tone. Built on top of a 7K-book dataset, it combines semantic vector search, zero-shot genre classification, and emotion analysis to deliver context-aware recommendations — all running locally without any paid API.

## What it does

The user types something like *"A story about forgiveness and second chances"*, picks a category (Fiction / Non-fiction / Children's / ...) and an emotional tone (Happy / Sad / Suspenseful / Angry / Surprising), and gets a grid of 16 book recommendations with covers, titles, authors, and short descriptions.

Under the hood:

1. **Text cleaning** — the raw Kaggle book dataset (~7K books) is loaded, missing fields handled, and short descriptions filtered out.
2. **Vector search** — book descriptions are embedded with `sentence-transformers/all-MiniLM-L6-v2` and indexed in a Chroma vector database. The user query is embedded with the same model and the top-K most similar books are retrieved by cosine similarity.
3. **Zero-shot classification** — every book is classified as *Fiction* / *Non-fiction* with `facebook/bart-large-mnli`, providing a category facet for filtering.
4. **Emotion extraction** — every book description is scored across 7 emotions (joy, sadness, anger, fear, surprise, disgust, neutral) using `j-hartmann/emotion-english-distilroberta-base`. The dominant scores are stored per book so the user can sort recommendations by tone.
5. **Desktop UI** — a `tkinter` window with a scrollable 4-column gallery. Cover images are loaded asynchronously in background threads so the interface never freezes.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Data | pandas, numpy |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs locally, no API key) |
| Vector DB | Chroma (`langchain-chroma`) |
| LLM pipeline | `langchain`, `langchain-huggingface`, `langchain-community` |
| Zero-shot classification | `facebook/bart-large-mnli` via `transformers` |
| Emotion analysis | `j-hartmann/emotion-english-distilroberta-base` via `transformers` |
| ML backend | `transformers`, `torch` |
| Desktop UI | `tkinter` + `Pillow` (covers) + `requests` (async download) |
| EDA / notebooks | `matplotlib`, `seaborn`, `jupyter` |
| Dataset | [7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) (via `kagglehub`) |

## Project structure

```
book-recommender/
├── data-exploration.ipynb     # Loads + cleans the Kaggle dataset → books_cleaned.csv
├── vector-search.ipynb        # Builds tagged_description.txt and demos Chroma similarity search
├── text-classification.ipynb  # Zero-shot Fiction/Non-fiction → books_with_categories.csv
├── sentiment-analysis.ipynb   # Per-book emotion scores → books_with_emotions.csv
├── tkinter-dashboard.py       # Desktop application (main entry point)
├── gradio-dashboard.py        # Original web version (kept for reference)
├── cover-not-found.jpg        # Fallback cover image
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebooks **in this order** to generate the data files:
   1. `data-exploration.ipynb` → produces `books_cleaned.csv`
   2. `vector-search.ipynb` → produces `tagged_description.txt`
   3. `text-classification.ipynb` → produces `books_with_categories.csv`
   4. `sentiment-analysis.ipynb` → produces `books_with_emotions.csv`

3. Launch the application:
   ```bash
   python tkinter-dashboard.py
   ```
   The first run downloads the embedding model (~80 MB) and builds the Chroma index in memory (30–60 sec). After that, queries are instant.

## Notes

- The original course version used OpenAI embeddings + Gradio. This fork swaps both: embeddings now run **locally** via Hugging Face (no API key, no cost), and the UI is a native desktop window via `tkinter`.
- All inference runs on CUDA by default. To use MPS, change `device="cuda"` to `device="mps"` in the classification notebooks.
- A `cover-not-found.jpg` fallback is shown when a book has no thumbnail URL.
