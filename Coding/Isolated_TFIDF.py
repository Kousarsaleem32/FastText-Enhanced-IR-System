
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download necessary resources for NLTK
download('punkt')
download('stopwords')

# Define file paths
documents_path = '/home/kausar/Assignment1/documents.csv'
queries_path = '/home/kausar/Assignment1/queries.csv'
ground_truth_path = '/home/kausar/Assignment1/ground_truth.csv'

# Load datasets
documents_df = pd.read_csv(documents_path)
queries_df = pd.read_csv(queries_path)
ground_truth_df = pd.read_csv(ground_truth_path)

# Print the first few rows of the loaded data
print("Documents DataFrame:")
print(documents_df.head())

print("\nQueries DataFrame:")
print(queries_df.head())

print("\nGround Truth DataFrame:")
print(ground_truth_df.head())

# Define stopwords and preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocesses text by lowercasing, tokenizing, and removing punctuation and stopwords."""
    if pd.notna(text):
        tokens = word_tokenize(str(text).lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    return ''

# Preprocess document and query texts
documents_df['processed_text'] = documents_df['text'].apply(preprocess_text)
queries_df['processed_text'] = queries_df['text'].apply(preprocess_text)

# Print the preprocessed text
print("\nPreprocessed Documents:")
print(documents_df[['text', 'processed_text']].head())

print("\nPreprocessed Queries:")
print(queries_df[['text', 'processed_text']].head())

# Create ground truth mapping
ground_truth = ground_truth_df.groupby('query_id')['doc_id'].apply(list).to_dict()

# Print ground truth mapping
print("\nGround Truth Mapping:")
for query_id, doc_ids in list(ground_truth.items())[:5]:  # Show the first 5 mappings
    print(f"Query ID {query_id}: Relevant Doc IDs: {doc_ids}")

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the TF-IDF vectorizer on document texts
tfidf_doc_matrix = tfidf_vectorizer.fit_transform(documents_df['processed_text'])

# Print TF-IDF shape and some feature names
print("\nTF-IDF Matrix Shape:", tfidf_doc_matrix.shape)
print("TF-IDF Features (first 10):", tfidf_vectorizer.get_feature_names_out()[:10])

def retrieve_top_10_tfidf(query, tfidf_doc_matrix, documents_df):
    """
    Retrieves the top 10 documents for a given query based on cosine similarity in TF-IDF space.
    Returns a list of document IDs and their similarity scores.
    """
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_doc_matrix).flatten()
    top_10_indices = np.argsort(cosine_similarities)[-10:][::-1]
    results = [(documents_df['doc_id'].iloc[i], cosine_similarities[i]) for i in top_10_indices]
    
    # Debugging: Print the ranked results for this query
    print(f"\nQuery: {query}")
    print("Top 10 Retrieved Documents:")
    for doc_id, score in results:
        print(f"Doc ID: {doc_id}, Score: {score:.4f}")
    
    return results

def calculate_average_precision(ranked_docs, relevant_docs):
    """
    Calculates Average Precision (AP) for a single query.
    """
    precision_at_k = []
    num_relevant_docs = 0

    for k, (doc_id, _) in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            num_relevant_docs += 1
            precision_at_k.append(num_relevant_docs / (k + 1))

    ap = np.mean(precision_at_k) if precision_at_k else 0.0

    # Debugging: Print AP calculation details
    print(f"\nRelevant Docs: {relevant_docs}")
    print(f"Ranked Docs: {[doc_id for doc_id, _ in ranked_docs]}")
    print(f"Precision at K: {precision_at_k}")
    print(f"Average Precision: {ap:.4f}")
    
    return ap

def evaluate_map(queries_df, ground_truth, tfidf_doc_matrix, documents_df):
    """
    Evaluates Mean Average Precision (MAP) over all queries.
    """
    mean_average_precision = 0.0

    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_text = row['processed_text']
        relevant_docs = ground_truth.get(query_id, [])
        ranked_docs = retrieve_top_10_tfidf(query_text, tfidf_doc_matrix, documents_df)
        average_precision = calculate_average_precision(ranked_docs, relevant_docs)
        mean_average_precision += average_precision

    map_score = mean_average_precision / len(ground_truth.keys())

    # Debugging: Print final MAP score
    print(f"\nMean Average Precision (MAP): {map_score:.4f}")
    
    return map_score

# Calculate MAP for TF-IDF
map_score = evaluate_map(queries_df, ground_truth, tfidf_doc_matrix, documents_df)
print(f"\nFinal Mean Average Precision (MAP) for TF-IDF: {map_score:.4f}")
