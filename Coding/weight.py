import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import string
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary resources for nltk
download('punkt')
download('stopwords')

# Define file paths
documents_path = 'C:/Information Retrieval 533/Repeat/Assignment1/documents.csv'
queries_path = 'C:/Information Retrieval 533/Repeat/Assignment1/queries.csv'
ground_truth_path = 'C:/Information Retrieval 533/Repeat/Assignment1/ground_truth.csv'

# Load datasets
documents_df = pd.read_csv(documents_path)
queries_df = pd.read_csv(queries_path)
ground_truth_df = pd.read_csv(ground_truth_path)

# Define stopwords and preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.notna(text):
        tokens = word_tokenize(str(text).lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in stop_words]
        return tokens
    else:
        return []

# Preprocess text in documents and queries
documents_df['processed_tokens'] = documents_df['text'].apply(preprocess_text)
queries_df['processed_tokens'] = queries_df['text'].apply(preprocess_text)

# Create ground truth mapping
ground_truth = {}
for row in ground_truth_df.itertuples(index=False):
    query_id = row.query_id
    doc_id = row.doc_id
    if query_id not in ground_truth:
        ground_truth[query_id] = []
    ground_truth[query_id].append(doc_id)

# Initialize TF-IDF
tfidf_vectorizer = TfidfVectorizer()
documents_df['processed_text'] = documents_df['processed_tokens'].apply(lambda x: ' '.join(x))
tfidf_doc_matrix = tfidf_vectorizer.fit_transform(documents_df['processed_text'])

# Initialize BM25
bm25 = BM25Okapi(documents_df['processed_tokens'].tolist())

# Normalize scores function
def normalize_scores(scores):
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()

# Compute embedding using a pre-trained or fine-tuned FastText model
def compute_embedding(tokens, model):
    embeddings = [model.wv[token] for token in tokens if token in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

# Function to calculate combined normalized scores
def retrieve_top_10_combined(query_tokens, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, model, weights):
    # TF-IDF Scores
    query_text = ' '.join(query_tokens)
    query_vec = tfidf_vectorizer.transform([query_text])
    tfidf_scores = cosine_similarity(query_vec, tfidf_doc_matrix).flatten()
    tfidf_scores = normalize_scores(tfidf_scores)

    # BM25 Scores
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_scores = normalize_scores(bm25_scores)

    # Embedding Scores
    query_embedding = compute_embedding(query_tokens, model)
    document_embeddings = documents_df['processed_tokens'].apply(lambda x: compute_embedding(x, model))
    embedding_scores = document_embeddings.apply(lambda x: np.dot(query_embedding, x) / (np.linalg.norm(query_embedding) * np.linalg.norm(x)))
    embedding_scores = normalize_scores(embedding_scores.values)

    # Combined Scores
    combined_scores = (weights['tfidf'] * tfidf_scores +
                       weights['bm25'] * bm25_scores +
                       weights['embedding'] * embedding_scores)

    # Get top 10 indices
    top_10_indices = np.argsort(combined_scores)[-10:][::-1]
    return [(documents_df['doc_id'].iloc[i], combined_scores[i]) for i in top_10_indices]

# Updated Average Precision (AP) calculation
def calculate_average_precision(ranked_docs, relevant_docs):
    precision_at_k = []
    num_relevant_docs_retrieved = 0
    for k, (doc_id, _) in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            num_relevant_docs_retrieved += 1
            precision_at_k.append(num_relevant_docs_retrieved / (k + 1))
    return np.mean(precision_at_k) if num_relevant_docs_retrieved > 0 else 0.0

# Evaluate MAP for all queries
def evaluate_map(queries_df, ground_truth, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, model, weights):
    mean_average_precision = 0.0
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_tokens = row['processed_tokens']
        relevant_docs = ground_truth.get(query_id, [])
        ranked_docs = retrieve_top_10_combined(query_tokens, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, model, weights)
        average_precision = calculate_average_precision(ranked_docs, relevant_docs)
        mean_average_precision += average_precision
    return mean_average_precision / len(ground_truth.keys())

# Load or fine-tune FastText model (provide your model path or train anew)
fine_tune_model_path = "cisi_fasttext_model.model"
print("Loading fine-tuned FastText model...")
fasttext_model = FastText.load(fine_tune_model_path)

# Experiment with different weight configurations
weight_configs = [
    {'tfidf': 0.3, 'bm25': 0.5, 'embedding': 0.2},
    {'tfidf': 0.4, 'bm25': 0.4, 'embedding': 0.2},
    {'tfidf': 0.2, 'bm25': 0.4, 'embedding': 0.4},
    {'tfidf': 0.3, 'bm25': 0.3, 'embedding': 0.4},
]

# Evaluate MAP for each weight configuration and store results
results = []
for i, weights in enumerate(weight_configs, start=1):
    print(f"\nEvaluating Weight Configuration {i}: TF-IDF={weights['tfidf']}, BM25={weights['bm25']}, Embedding={weights['embedding']}")
    map_score = evaluate_map(queries_df, ground_truth, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, fasttext_model, weights)
    print(f"MAP for Configuration {i}: {map_score:.4f}")
    results.append((i, weights, map_score))

# Print a summary of all configurations and MAP scores
print("\nSummary of MAP Scores for All Weight Configurations:")
for config_id, weights, map_score in results:
    print(f"Configuration {config_id}: TF-IDF={weights['tfidf']}, BM25={weights['bm25']}, Embedding={weights['embedding']} -> MAP: {map_score:.4f}")
