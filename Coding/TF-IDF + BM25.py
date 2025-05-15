import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Initialize TF-IDF
tfidf_vectorizer = TfidfVectorizer()
documents_df['processed_text'] = documents_df['processed_tokens'].apply(lambda x: ' '.join(x))
tfidf_doc_matrix = tfidf_vectorizer.fit_transform(documents_df['processed_text'])

# Initialize BM25
bm25 = BM25Okapi(documents_df['processed_tokens'].tolist())

# Function to retrieve top 10 documents based on combined TF-IDF and BM25 scores
def retrieve_top_10_combined(query_tokens, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, weights):
    # TF-IDF Scores
    query_text = ' '.join(query_tokens)
    query_vec = tfidf_vectorizer.transform([query_text])
    tfidf_scores = cosine_similarity(query_vec, tfidf_doc_matrix).flatten()

    # BM25 Scores
    bm25_scores = bm25.get_scores(query_tokens)

    # Combined Scores
    combined_scores = weights['tfidf'] * tfidf_scores + weights['bm25'] * bm25_scores

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
def evaluate_map(queries_df, ground_truth, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, weights):
    mean_average_precision = 0.0
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_tokens = row['processed_tokens']
        relevant_docs = ground_truth.get(query_id, [])
        ranked_docs = retrieve_top_10_combined(query_tokens, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, weights)
        average_precision = calculate_average_precision(ranked_docs, relevant_docs)
        mean_average_precision += average_precision
    return mean_average_precision / len(ground_truth.keys())

# Define weights for TF-IDF and BM25
weights = {'tfidf': 0.5, 'bm25': 0.5}

# Calculate MAP for combined TF-IDF + BM25
map_score = evaluate_map(queries_df, ground_truth, tfidf_vectorizer, tfidf_doc_matrix, bm25, documents_df, weights)
print(f"Mean Average Precision (MAP) for Combined TF-IDF + BM25: {map_score:.4f}")
