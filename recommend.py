import pickle
import numpy as np

print("Loading recommendation model...")

try:
    # Load model and encoders
    with open("recommender_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    
    with open("post_encoder.pkl", "rb") as f:
        post_encoder = pickle.load(f)
    
    print("✓ Model loaded successfully!")
    print(f"  • Users: {len(user_encoder.classes_)}")
    print(f"  • Posts: {len(post_encoder.classes_)}")
    print(f"  • User factors shape: {model.user_factors.shape}")
    print(f"  • Item factors shape: {model.item_factors.shape}")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

def recommend_for_user(user_id, top_k=10):
    """
    Get recommendations for a specific user
    
    Args:
        user_id: Original user ID (must exist in training data)
        top_k: Number of recommendations to return
    
    Returns:
        list: Recommended post IDs with scores
    """
    try:
        # Check if user exists
        if user_id not in user_encoder.classes_:
            print(f"⚠ User {user_id} not found in training data")
            return []
        
        # Encode user
        user_idx = user_encoder.transform([user_id])[0]
        
        # Calculate user-item scores
        user_vector = model.user_factors[user_idx]
        scores = user_vector @ model.item_factors.T
        
        # Get top K indices
        top_indices = np.argsort(-scores)[:top_k]
        
        # Get scores for top indices
        top_scores = scores[top_indices]
        
        # Decode to original post IDs
        post_ids = post_encoder.inverse_transform(top_indices)
        
        # Return as list of (post_id, score) tuples
        return list(zip(post_ids, top_scores))
        
    except Exception as e:
        print(f"✗ Error getting recommendations: {e}")
        return []

def get_similar_posts(post_id, top_k=5):
    """
    Find posts similar to a given post
    
    Args:
        post_id: Original post ID
        top_k: Number of similar posts
    
    Returns:
        list: Similar post IDs with similarity scores
    """
    try:
        # Check if post exists
        if post_id not in post_encoder.classes_:
            print(f"⚠ Post {post_id} not found in training data")
            return []
        
        # Encode post
        post_idx = post_encoder.transform([post_id])[0]
        
        # Get similar items
        similar = model.similar_items(post_idx, N=top_k+1)
        
        # Skip the first one (it's the post itself)
        similar_indices = similar[0][1:top_k+1]
        similarity_scores = similar[1][1:top_k+1]
        
        # Decode to original IDs
        similar_ids = post_encoder.inverse_transform(similar_indices)
        
        return list(zip(similar_ids, similarity_scores))
        
    except Exception as e:
        print(f"✗ Error finding similar posts: {e}")
        return []

def get_user_history(user_id, limit=5):
    """
    Get posts that a user has interacted with
    """
    try:
        if user_id not in user_encoder.classes_:
            return []
        
        user_idx = user_encoder.transform([user_id])[0]
        
        # This would require loading the original data
        # For now, just return an empty list
        return []
        
    except:
        return []

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TESTING RECOMMENDATIONS")
    print("=" * 50)
    
    # Get sample users and posts
    sample_users = user_encoder.classes_[:3]
    sample_posts = post_encoder.classes_[:3]
    
    print(f"\nSample users: {sample_users}")
    print(f"Sample posts: {sample_posts}")
    
    # Test user recommendations
    print("\n1. User Recommendations:")
    for user_id in sample_users:
        print(f"\n  For User {user_id}:")
        recommendations = recommend_for_user(user_id, top_k=3)
        
        if recommendations:
            for i, (post_id, score) in enumerate(recommendations, 1):
                print(f"    {i}. Post {post_id} (score: {score:.3f})")
        else:
            print("    No recommendations available")
    
    # Test similar posts
    print("\n2. Similar Posts:")
    for post_id in sample_posts:
        print(f"\n  Posts similar to Post {post_id}:")
        similar = get_similar_posts(post_id, top_k=3)
        
        if similar:
            for i, (similar_id, similarity) in enumerate(similar, 1):
                print(f"    {i}. Post {similar_id} (similarity: {similarity:.3f})")
        else:
            print("    No similar posts found")
    
    # Test with custom user ID
    print("\n3. Test with Custom User ID:")
    test_user = 12  # Change this to a user ID in your data
    print(f"\n  Testing with User {test_user}:")
    if test_user in user_encoder.classes_:
        recs = recommend_for_user(test_user, top_k=5)
        if recs:
            for i, (post_id, score) in enumerate(recs, 1):
                print(f"    {i}. Post {post_id} (score: {score:.3f})")
        else:
            print("    No recommendations")
    else:
        print(f"    User {test_user} not in training data")
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)
    print("\nUsage:")
    print("  from recommend import recommend_for_user, get_similar_posts")
    print("  recs = recommend_for_user(user_id=123, top_k=10)")
    print("  similar = get_similar_posts(post_id=456, top_k=5)")