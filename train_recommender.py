import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import pickle
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
DATA_PATH = "user_post_interactions_30k.csv"
MODEL_PATH = "recommender_model.pkl"
USER_ENCODER_PATH = "user_encoder.pkl"
POST_ENCODER_PATH = "post_encoder.pkl"

# Model hyperparameters
FACTORS = 64
REGULARIZATION = 0.01
ITERATIONS = 20
RANDOM_STATE = 42
USE_GPU = False
TEST_SIZE = 0.2  # 20% for testing

# =========================
# 1. Load and validate dataset
# =========================
print("=" * 50)
print("RECOMMENDER SYSTEM TRAINING WITH EVALUATION")
print("=" * 50)

print(f"\n[1/6] Loading data from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df):,} interactions")
    
    # Check required columns
    required_columns = ['user_id', 'post_id', 'interaction_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    
    print(f"✓ Data has required columns: {df.columns.tolist()}")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# Show data stats
print(f"\n[2/6] Data Statistics:")
print(f"  • Unique users: {df['user_id'].nunique():,}")
print(f"  • Unique posts: {df['post_id'].nunique():,}")
print(f"  • Interactions: {len(df):,}")
print(f"  • Avg interactions per user: {len(df) / df['user_id'].nunique():.1f}")
print(f"  • Avg interactions per post: {len(df) / df['post_id'].nunique():.1f}")
print(f"  • Interaction score range: {df['interaction_score'].min()} to {df['interaction_score'].max()}")
print(f"  • Interaction score mean: {df['interaction_score'].mean():.2f}")

# =========================
# 2. Encode IDs
# =========================
print(f"\n[3/6] Encoding user and post IDs...")

# Create encoders
user_encoder = LabelEncoder()
post_encoder = LabelEncoder()

# Fit and transform
df["user_idx"] = user_encoder.fit_transform(df["user_id"])
df["post_idx"] = post_encoder.fit_transform(df["post_id"])

num_users = len(user_encoder.classes_)
num_posts = len(post_encoder.classes_)

print(f"✓ Encoding complete:")
print(f"  • Encoded users: {num_users:,}")
print(f"  • Encoded posts: {num_posts:,}")

# =========================
# 3. Train-Test Split
# =========================
print(f"\n[4/6] Creating train-test split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} split)...")

# Create a copy for train-test split
df_copy = df.copy()

# Group by user for better split
user_groups = df_copy.groupby('user_idx')

train_data = []
test_data = []

for user_idx, group in user_groups:
    if len(group) > 1:
        # Split each user's interactions
        train, test = train_test_split(
            group, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE
        )
        train_data.append(train)
        test_data.append(test)
    else:
        # Users with only one interaction go to train
        train_data.append(group)

train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

print(f"✓ Split complete:")
print(f"  • Training set: {len(train_df):,} interactions ({len(train_df)/len(df)*100:.1f}%)")
print(f"  • Test set: {len(test_df):,} interactions ({len(test_df)/len(df)*100:.1f}%)")
print(f"  • Test users: {test_df['user_idx'].nunique():,}")
print(f"  • Test posts: {test_df['post_idx'].nunique():,}")

# =========================
# 4. Build interaction matrices
# =========================
print(f"\n[5/6] Building interaction matrices...")

start_time = time.time()

# Build training matrix
train_matrix = csr_matrix(
    (
        train_df["interaction_score"].values.astype(np.float32),
        (train_df["user_idx"].values, train_df["post_idx"].values)
    ),
    shape=(num_users, num_posts)
)

# Build test matrix (for evaluation)
test_matrix = csr_matrix(
    (
        test_df["interaction_score"].values.astype(np.float32),
        (test_df["user_idx"].values, test_df["post_idx"].values)
    ),
    shape=(num_users, num_posts)
)

# Apply BM25 weighting to training matrix
print("  Applying BM25 weighting...")
weighted_matrix = bm25_weight(train_matrix, K1=100, B=0.8)

matrix_time = time.time() - start_time
print(f"✓ Matrices created in {matrix_time:.2f} seconds")
print(f"  • Train matrix shape: {weighted_matrix.shape}")
print(f"  • Train non-zero entries: {weighted_matrix.nnz:,}")
print(f"  • Test matrix shape: {test_matrix.shape}")
print(f"  • Test non-zero entries: {test_matrix.nnz:,}")

# =========================
# 5. Train ALS Model
# =========================
print(f"\n[6/6] Training ALS model...")

# Initialize model
model = AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    random_state=RANDOM_STATE,
    use_gpu=USE_GPU,
    calculate_training_loss=True
)

print(f"Model configuration:")
print(f"  • Factors: {FACTORS}")
print(f"  • Regularization: {REGULARIZATION}")
print(f"  • Iterations: {ITERATIONS}")
print(f"  • Using GPU: {USE_GPU}")

# Train model
start_time = time.time()
model.fit(weighted_matrix)
training_time = time.time() - start_time

print(f"✓ Model trained in {training_time:.2f} seconds")

# =========================
# 6. Evaluate Model
# =========================
print(f"\n[7/6] Evaluating model on test set...")

def evaluate_model_simple(model, test_df, train_df, k=10):
    """
    Simple evaluation using precision@k and recall@k
    
    Args:
        model: Trained ALS model
        test_df: Test interactions DataFrame
        train_df: Training interactions DataFrame
        k: Number of recommendations to consider
    """
    # Convert to dictionaries for faster lookup
    user_test_items = test_df.groupby('user_idx')['post_idx'].apply(set).to_dict()
    user_train_items = train_df.groupby('user_idx')['post_idx'].apply(set).to_dict()
    
    total_precision = 0.0
    total_recall = 0.0
    total_users = 0
    
    # Get test users
    test_users = list(user_test_items.keys())
    
    # Evaluate on first 100 users for speed
    eval_users = test_users[:min(100, len(test_users))]
    
    for user_idx in eval_users:
        test_items = user_test_items.get(user_idx, set())
        train_items = user_train_items.get(user_idx, set())
        
        if not test_items:
            continue
        
        # Get recommendations (filter out training items)
        try:
            # Create user vector
            user_vector = model.user_factors[user_idx]
            
            # Calculate scores for all items
            scores = user_vector @ model.item_factors.T
            
            # Filter out items already seen in training
            all_items = np.arange(num_posts)
            unseen_items = [i for i in all_items if i not in train_items]
            
            if not unseen_items:
                continue
                
            # Get scores for unseen items only
            unseen_scores = scores[unseen_items]
            
            # Get top k unseen items
            top_indices = np.argsort(-unseen_scores)[:k]
            rec_items = set([unseen_items[i] for i in top_indices])
            
            # Calculate metrics
            if rec_items:
                relevant_recommended = len(rec_items.intersection(test_items))
                precision = relevant_recommended / len(rec_items)
                recall = relevant_recommended / len(test_items)
                
                total_precision += precision
                total_recall += recall
                total_users += 1
                
        except Exception as e:
            continue
    
    # Calculate averages
    if total_users > 0:
        avg_precision = total_precision / total_users
        avg_recall = total_recall / total_users
    else:
        avg_precision = avg_recall = 0
    
    return {
        'precision@k': avg_precision,
        'recall@k': avg_recall,
        'users_evaluated': total_users
    }

# Evaluate with different k values
print("\nEvaluation Results:")
print("-" * 40)

for k in [5, 10, 20]:
    metrics = evaluate_model_simple(model, test_df, train_df, k=k)
    print(f"\nk = {k}:")
    print(f"  • Precision@{k}: {metrics['precision@k']:.4f}")
    print(f"  • Recall@{k}: {metrics['recall@k']:.4f}")
    print(f"  • Users evaluated: {metrics['users_evaluated']}")

# Test sample predictions
print(f"\n[INFO] Sample predictions:")
try:
    # Get a test user with interactions
    test_users = test_df['user_idx'].unique()
    if len(test_users) > 0:
        test_user_idx = test_users[0]
        test_user_id = user_encoder.inverse_transform([test_user_idx])[0]
        
        # Get user's test items
        user_test_items = test_df[test_df['user_idx'] == test_user_idx]['post_idx'].values
        test_item_ids = post_encoder.inverse_transform(user_test_items)
        
        # Get user's train items (to filter out)
        user_train_items = train_df[train_df['user_idx'] == test_user_idx]['post_idx'].values
        
        # Calculate recommendations manually
        user_vector = model.user_factors[test_user_idx]
        scores = user_vector @ model.item_factors.T
        
        # Filter out training items
        all_items = np.arange(num_posts)
        train_set = set(user_train_items)
        unseen_items = [i for i in all_items if i not in train_set]
        
        if unseen_items:
            unseen_scores = scores[unseen_items]
            top_indices = np.argsort(-unseen_scores)[:5]
            rec_indices = [unseen_items[i] for i in top_indices]
            rec_scores = unseen_scores[top_indices]
            
            rec_ids = post_encoder.inverse_transform(rec_indices)
            
            print(f"\n  Test User {test_user_id}:")
            print(f"    Training interactions: {len(user_train_items)} items")
            print(f"    Test interactions: {len(user_test_items)} items")
            print(f"    Recommendations (top 5):")
            
            for i, (post_id, score) in enumerate(zip(rec_ids, rec_scores)):
                # Check if this post is in test set
                is_correct = post_id in test_item_ids
                mark = "✓" if is_correct else " "
                print(f"      {mark} {i+1}. Post {post_id} (score: {score:.3f})")
            
            # Calculate hits
            hits = set(rec_ids).intersection(set(test_item_ids))
            if hits:
                print(f"    ✓ Correct predictions: {len(hits)}/{len(rec_ids)} hits")
            else:
                print(f"    ✗ No correct predictions")
                
except Exception as e:
    print(f"  Could not generate sample predictions: {e}")

# =========================
# 7. Save artifacts
# =========================
print(f"\n[8/6] Saving artifacts...")

try:
    # Save main artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {MODEL_PATH}")
    
    with open(USER_ENCODER_PATH, "wb") as f:
        pickle.dump(user_encoder, f)
    print(f"✓ User encoder saved to: {USER_ENCODER_PATH}")
    
    with open(POST_ENCODER_PATH, "wb") as f:
        pickle.dump(post_encoder, f)
    print(f"✓ Post encoder saved to: {POST_ENCODER_PATH}")
    
    # Save training and test data info
    training_info = {
        'num_users': num_users,
        'num_posts': num_posts,
        'train_interactions': len(train_df),
        'test_interactions': len(test_df),
        'test_users': test_df['user_idx'].nunique(),
        'test_posts': test_df['post_idx'].nunique(),
        'factors': FACTORS,
        'regularization': REGULARIZATION,
        'iterations': ITERATIONS,
        'training_time': training_time,
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'train_matrix_shape': weighted_matrix.shape,
        'train_matrix_nnz': weighted_matrix.nnz,
        'test_matrix_shape': test_matrix.shape,
        'test_matrix_nnz': test_matrix.nnz
    }
    
    with open("training_info.pkl", "wb") as f:
        pickle.dump(training_info, f)
    print(f"✓ Training info saved to: training_info.pkl")
    
    # Save test data for later evaluation
    test_data_info = {
        'test_df': test_df[['user_idx', 'post_idx', 'interaction_score']].copy(),
        'train_df': train_df[['user_idx', 'post_idx', 'interaction_score']].copy()
    }
    
    with open("test_data.pkl", "wb") as f:
        pickle.dump(test_data_info, f)
    print(f"✓ Test data saved to: test_data.pkl")
    
except Exception as e:
    print(f"✗ Error saving artifacts: {e}")
    exit(1)

# =========================
# Summary
# =========================
print(f"\n" + "=" * 50)
print("TRAINING AND EVALUATION COMPLETE!")
print("=" * 50)
print(f"\nSummary:")
print(f"  • Total users: {num_users:,}")
print(f"  • Total posts: {num_posts:,}")
print(f"  • Training interactions: {len(train_df):,}")
print(f"  • Test interactions: {len(test_df):,}")
print(f"  • Model factors: {FACTORS}")
print(f"  • Training time: {training_time:.2f}s")

# Final evaluation
print(f"\nFinal Evaluation Metrics:")
metrics_10 = evaluate_model_simple(model, test_df, train_df, k=10)
print(f"  • Precision@10: {metrics_10['precision@k']:.4f}")
print(f"  • Recall@10: {metrics_10['recall@k']:.4f}")
print(f"  • F1-Score@10: {2 * metrics_10['precision@k'] * metrics_10['recall@k'] / (metrics_10['precision@k'] + metrics_10['recall@k'] + 1e-10):.4f}")

print(f"\nFiles created:")
print(f"  1. {MODEL_PATH} - Trained ALS model")
print(f"  2. {USER_ENCODER_PATH} - User ID encoder")
print(f"  3. {POST_ENCODER_PATH} - Post ID encoder")
print(f"  4. training_info.pkl - Training metadata")
print(f"  5. test_data.pkl - Test data for evaluation")
print(f"\nTo test recommendations, run:")
print(f"  python recommend.py")
print("=" * 50)