import pandas as pd
import numpy as np
import fasttext.util
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import string

# Download necessary resources for nltk
download('punkt')
download('stopwords')

# Define file paths
documents_path = '/home/kausar/Assignment1/documents.csv'
queries_path = '/home/kausar/Assignment1/queries.csv'
ground_truth_path = '/home/kausar/Assignment1/ground_truth.csv'

# Load each dataset
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

# Preprocess document and query texts
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

# Initialize BM25
bm25 = BM25Okapi(documents_df['processed_tokens'].tolist())

# Download and load pre-trained FastText model
fasttext.util.download_model('en', if_exists='ignore')  # Downloads 'cc.en.300.bin'
fasttext_model_path = '/home/kausar/Assignment1/cc.en.300.bin'
fasttext_model = fasttext.load_model(fasttext_model_path)

# Function to compute the average embedding for a document or query
def compute_embedding(tokens, model):
    embeddings = [model[token] for token in tokens if token in model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(300)  # Assume 300-dimensional embeddings for FastText

# Function to retrieve top 10 documents based on combined BM25 and Embedding scores
def retrieve_top_10_combined(query_tokens, bm25, documents_df, model, weights):
    # BM25 Scores
    bm25_scores = bm25.get_scores(query_tokens)

    # Embedding Scores
    query_embedding = compute_embedding(query_tokens, model)
    document_embeddings = documents_df['processed_tokens'].apply(lambda x: compute_embedding(x, model))
    embedding_scores = document_embeddings.apply(lambda x: cosine_similarity([query_embedding], [x]).flatten()[0])

    # Combined Scores
    combined_scores = weights['bm25'] * bm25_scores + weights['embedding'] * embedding_scores

    # Get top 10 indices
    top_10_indices = np.argsort(combined_scores)[-10:][::-1]
    return [(documents_df['doc_id'].iloc[i], combined_scores[i]) for i in top_10_indices]

# Function to calculate Average Precision (AP) for a single query
def calculate_average_precision(ranked_docs, relevant_docs):
    precision_at_k = []
    num_relevant_docs = 0
    for k, (doc_id, _) in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            num_relevant_docs += 1
            precision_at_k.append(num_relevant_docs / (k + 1))
    return np.mean(precision_at_k) if precision_at_k else 0.0

# Function to evaluate MAP for all queries
def evaluate_map(queries_df, ground_truth, bm25, documents_df, model, weights):
    mean_average_precision = 0.0
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_tokens = row['processed_tokens']
        relevant_docs = ground_truth.get(query_id, [])
        ranked_docs = retrieve_top_10_combined(query_tokens, bm25, documents_df, model, weights)
        average_precision = calculate_average_precision(ranked_docs, relevant_docs)
        mean_average_precision += average_precision
    return mean_average_precision / len(ground_truth.keys())

# Define weights for BM25 and Embedding
weights = {'bm25': 0.5, 'embedding': 0.5}

# Calculate MAP for combined BM25 + Embedding
map_score = evaluate_map(queries_df, ground_truth, bm25, documents_df, fasttext_model, weights)
print(f"Mean Average Precision (MAP) for Combined BM25 + Embedding: {map_score:.4f}")
