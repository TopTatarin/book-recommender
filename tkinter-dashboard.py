import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
from io import BytesIO
import threading
import os

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, hf_embeddings)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


THUMB_W, THUMB_H = 150, 220
GRID_COLS = 4
FALLBACK_PATH = "cover-not-found.jpg"

_image_refs = []


def _load_fallback_image():
    if os.path.exists(FALLBACK_PATH):
        try:
            img = Image.open(FALLBACK_PATH).convert("RGB")
            img.thumbnail((THUMB_W, THUMB_H))
            return ImageTk.PhotoImage(img)
        except Exception:
            pass
    placeholder = Image.new("RGB", (THUMB_W, THUMB_H), color=(220, 220, 220))
    return ImageTk.PhotoImage(placeholder)


def _fetch_image(url: str):
    try:
        if not url or url == FALLBACK_PATH or not str(url).startswith("http"):
            return None
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.thumbnail((THUMB_W, THUMB_H))
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _load_image_async(url: str, label: tk.Label, fallback_image):
    def worker():
        img = _fetch_image(url)
        final = img if img is not None else fallback_image

        def apply():
            if label.winfo_exists():
                label.configure(image=final)
                label.image = final
                _image_refs.append(final)
        try:
            label.after(0, apply)
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


def render_gallery(results):
    for child in gallery_frame.winfo_children():
        child.destroy()
    _image_refs.clear()

    fallback_image = _load_fallback_image()
    _image_refs.append(fallback_image)

    if not results:
        empty = ttk.Label(gallery_frame, text="No recommendations found.")
        empty.grid(row=0, column=0, padx=10, pady=10)
        return

    for idx, (thumb_url, caption) in enumerate(results):
        row = idx // GRID_COLS
        col = idx % GRID_COLS

        card = ttk.Frame(gallery_frame, padding=8)
        card.grid(row=row, column=col, padx=8, pady=8, sticky="n")

        img_label = tk.Label(card, image=fallback_image, width=THUMB_W, height=THUMB_H)
        img_label.image = fallback_image
        img_label.pack()

        text_label = ttk.Label(
            card,
            text=caption,
            wraplength=THUMB_W,
            justify="left",
            anchor="w",
        )
        text_label.pack(pady=(6, 0), fill="x")

        _load_image_async(thumb_url, img_label, fallback_image)

    gallery_frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))


def on_submit():
    query = query_var.get().strip()
    if not query:
        return
    category = category_var.get()
    tone = tone_var.get()
    submit_button.configure(state="disabled", text="Searching...")
    root.update_idletasks()
    try:
        results = recommend_books(query, category, tone)
        render_gallery(results)
    finally:
        submit_button.configure(state="normal", text="Find recommendations")


root = tk.Tk()
root.title("Semantic book recommender")
root.geometry("780x780")

title = ttk.Label(root, text="Semantic book recommender", font=("Segoe UI", 16, "bold"))
title.pack(pady=(12, 8))

controls = ttk.Frame(root, padding=10)
controls.pack(fill="x")

query_var = tk.StringVar()
category_var = tk.StringVar(value="All")
tone_var = tk.StringVar(value="All")

ttk.Label(controls, text="Please enter a description of a book:").grid(row=0, column=0, sticky="w")
query_entry = ttk.Entry(controls, textvariable=query_var, width=50)
query_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(2, 8))
query_entry.insert(0, "")

ttk.Label(controls, text="Select a category:").grid(row=2, column=0, sticky="w")
category_combo = ttk.Combobox(controls, textvariable=category_var, values=categories, state="readonly")
category_combo.grid(row=3, column=0, sticky="we", padx=(0, 6), pady=(2, 8))

ttk.Label(controls, text="Select an emotional tone:").grid(row=2, column=1, sticky="w")
tone_combo = ttk.Combobox(controls, textvariable=tone_var, values=tones, state="readonly")
tone_combo.grid(row=3, column=1, sticky="we", padx=(6, 0), pady=(2, 8))

controls.columnconfigure(0, weight=1)
controls.columnconfigure(1, weight=1)

submit_button = ttk.Button(controls, text="Find recommendations", command=on_submit)
submit_button.grid(row=4, column=0, columnspan=2, pady=(4, 0))

ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=8)
ttk.Label(root, text="Recommendations", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14)

gallery_container = ttk.Frame(root)
gallery_container.pack(fill="both", expand=True, padx=10, pady=10)

canvas = tk.Canvas(gallery_container, highlightthickness=0)
scrollbar = ttk.Scrollbar(gallery_container, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

gallery_frame = ttk.Frame(canvas)
canvas_window = canvas.create_window((0, 0), window=gallery_frame, anchor="nw")


def _on_gallery_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


def _on_canvas_configure(event):
    canvas.itemconfigure(canvas_window, width=event.width)


def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


gallery_frame.bind("<Configure>", _on_gallery_configure)
canvas.bind("<Configure>", _on_canvas_configure)
canvas.bind_all("<MouseWheel>", _on_mousewheel)


if __name__ == "__main__":
    root.mainloop()
