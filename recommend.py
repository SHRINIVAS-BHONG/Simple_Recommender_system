import numpy as np

# -----------------------------
# Load embeddings
# -----------------------------
post_embeddings = np.load("post_embeddings.npy", allow_pickle=True).item()
user_embeddings = np.load("user_embeddings.npy", allow_pickle=True).item()

# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_posts(user_id, top_k=5):
    user_vec = user_embeddings[user_id]
    scores = []

    for post_id, post_vec in post_embeddings.items():
        score = cosine_similarity(user_vec, post_vec)
        scores.append((post_id, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# -----------------------------
# Run recommendations
# -----------------------------
print("\nRecommendations for user_id=7:")
print(recommend_posts(user_id=7))

print("\nRecommendations for user_id=3:")
print(recommend_posts(user_id=3))
