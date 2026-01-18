import json
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load data
# -----------------------------
with open("data.json", "r") as f:
    data = json.load(f)

users = data["users"]
posts = data["posts"]
interactions = data["interactions"]

# -----------------------------
# Load text embedding model
# -----------------------------
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Engagement score
# -----------------------------
def engagement_score(post):
    return (
        1.0 * post["likes"] +
        2.0 * post["comments"] +
        3.0 * post["shares"]
    )

# -----------------------------
# Build Post Embeddings
# -----------------------------
post_embeddings = {}

for post in posts:
    text_emb = text_model.encode(post["text"])
    eng = engagement_score(post)
    eng_emb = np.array([eng], dtype=np.float32)

    final_embedding = np.concatenate([text_emb, eng_emb])
    post_embeddings[post["post_id"]] = final_embedding

print("Post embeddings created")

# -----------------------------
# Build User Embeddings
# -----------------------------
embedding_dim = len(next(iter(post_embeddings.values())))
user_embeddings = {}

for user in users:
    user_vec = np.zeros(embedding_dim, dtype=np.float32)

    for inter in interactions:
        if inter["user_id"] == user["user_id"]:
            user_vec += post_embeddings[inter["post_id"]]

    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    user_embeddings[user["user_id"]] = user_vec

print("User embeddings created")

# -----------------------------
# Save embeddings
# -----------------------------
np.save("post_embeddings.npy", post_embeddings)
np.save("user_embeddings.npy", user_embeddings)

print("Training complete. Embeddings saved.")
