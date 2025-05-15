import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
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

# Function to retrieve top 10 documents for a query based on BM25
def retrieve_top_10_bm25(query_tokens, bm25, documents_df):
    # Get BM25 scores for all documents
    scores = bm25.get_scores(query_tokens)
    # Get indices of the top 10 documents
    top_10_indices = np.argsort(scores)[-10:][::-1]
    # Retrieve document IDs and scores
    return [(documents_df['doc_id'].iloc[i], scores[i]) for i in top_10_indices]

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
def evaluate_map(queries_df, ground_truth, bm25, documents_df):
    mean_average_precision = 0.0
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_tokens = row['processed_tokens']
        relevant_docs = ground_truth.get(query_id, [])
        ranked_docs = retrieve_top_10_bm25(query_tokens, bm25, documents_df)
        average_precision = calculate_average_precision(ranked_docs, relevant_docs)
        mean_average_precision += average_precision
    return mean_average_precision / len(ground_truth.keys())

# Calculate MAP for BM25
map_score = evaluate_map(queries_df, ground_truth, bm25, documents_df)
print(f"Mean Average Precision (MAP) for BM25: {map_score:.4f}")
