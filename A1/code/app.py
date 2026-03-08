from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# -----------------------------
# Load corpus embeddings
# -----------------------------
MODEL_DIR = "../model"

with open(os.path.join(MODEL_DIR, "embed_glove.pkl"), "rb") as f:
    embeddings = pickle.load(f)   # {word: vector}

# -----------------------------
# Dot product similarity
# -----------------------------
def dot_product(a, b):
    return float(np.dot(a, b))


def top_10_similar_contexts(query, embeddings, top_n=10):
    if query not in embeddings:
        return [(0.0, "Word not found in corpus")]

    query_vec = embeddings[query]
    scores = []

    for word, vec in embeddings.items():
        score = dot_product(query_vec, vec)
        scores.append((score, word))

    # sort by dot product (descending)
    scores = sorted(scores, reverse=True)

    # remove query itself
    scores = [(round(score, 4), word) for score, word in scores if word != query]

    return scores[:top_n]

# -----------------------------
# Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query").lower().strip()
        results = top_10_similar_contexts(query, embeddings)

    return render_template(
        "index.html",
        query=query,
        results=results
    )

if __name__ == "__main__":
    app.run(debug=True)
