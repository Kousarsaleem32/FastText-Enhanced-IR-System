import pandas as pd
import numpy as np
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import string

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

# Preprocess the text in documents and queries
documents_df['processed_tokens'] = documents_df['text'].apply(preprocess_text)
queries_df['processed_tokens'] = queries_df['text'].apply(preprocess_text)

# Combine tokens from documents and queries
all_tokens = documents_df['processed_tokens'].tolist() + queries_df['processed_tokens'].tolist()

# Train FastText model using Gensim
print("Training FastText model...")
fasttext_model = FastText(sentences=all_tokens, vector_size=100, window=5, min_count=1, epochs=10)

# Save the trained model
fasttext_model_path = 'cisi_fasttext_model.model'
fasttext_model.save(fasttext_model_path)
print(f"FastText model saved to {fasttext_model_path}")

# Function to compute the average embedding for a document or query
def compute_embedding(tokens, model):
    embeddings = [model.wv[token] for token in tokens if token in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

# Create ground truth mapping
ground_truth = {}
for row in ground_truth_df.itertuples(index=False):
    query_id = row.query_id
    doc_id = row.doc_id
    if query_id not in ground_truth:
        ground_truth[query_id] = []
    ground_truth[query_id].append(doc_id)

# Function to retrieve top 10 documents based on trained FastText embeddings
def retrieve_top_10_fasttext(query_tokens, documents_df, model):
    query_embedding = compute_embedding(query_tokens, model)
    document_embeddings = documents_df['processed_tokens'].apply(lambda x: compute_embedding(x, model))
    similarities = document_embeddings.apply(lambda x: np.dot(query_embedding, x) / (np.linalg.norm(query_embedding) * np.linalg.norm(x)))
    top_10_indices = similarities.nlargest(10).index
    return [(documents_df['doc_id'].iloc[i], similarities.iloc[i]) for i in top_10_indices]

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
def evaluate_map(queries_df, ground_truth, documents_df, model):
    mean_average_precision = 0.0
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_tokens = row['processed_tokens']
        relevant_docs = ground_truth.get(query_id, [])
        ranked_docs = retrieve_top_10_fasttext(query_tokens, documents_df, model)
        average_precision = calculate_average_precision(ranked_docs, relevant_docs)
        mean_average_precision += average_precision
    return mean_average_precision / len(ground_truth.keys())

# Evaluate MAP for the trained FastText model
map_score = evaluate_map(queries_df, ground_truth, documents_df, fasttext_model)
print(f"Mean Average Precision (MAP) for Trained FastText Model on CISI Dataset: {map_score:.4f}")
