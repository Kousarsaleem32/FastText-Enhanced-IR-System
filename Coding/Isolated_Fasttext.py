import pandas as pd
import numpy as np
import fasttext
import fasttext.util
import re
from nltk.corpus import stopwords
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import nltk
nltk.download('punkt_tab')

# download('punkt')
download('stopwords')


# Functions for text preprocessing
def preprocess_text(text):
    if pd.notna(text):
        tokens = word_tokenize(str(text).lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    else:
        return ''

def compute_embedding(model, text):
    tokens = word_tokenize(text.lower())

    if not tokens:
        return np.zeros(300)  # Assuming 300 dimensions for the FastText model

    valid_tokens = [token for token in tokens if token in model]
    if not valid_tokens:
        return np.zeros(300)

    return np.mean([model[token] for token in valid_tokens], axis=0)

def retrieve_documents(query, documents, word2vec_model, weights):
    query_embedding = compute_embedding(word2vec_model, preprocess_text(query))
    document_embeddings = [compute_embedding(word2vec_model, preprocess_text(doc)) for doc in documents]
    embedding_scores = cosine_similarity([query_embedding], document_embeddings).flatten()

    ranked_indices = np.argsort(embedding_scores)[::-1][:10]
    ranked_documents = [documents[i] for i in ranked_indices]

    return ranked_indices

def calculate_average_precision(ranked_indices, relevant_docs):
    precision_at_k = []
    num_relevant_docs = 0

    for i, idx in enumerate(ranked_indices):
        if idx in relevant_docs:
            num_relevant_docs += 1
            precision_at_k.append(num_relevant_docs / (i + 1))

    if not precision_at_k:
        return 0.0

    return np.mean(precision_at_k)

def evaluate_system(queries, ground_truth, documents, word2vec_model, weights):
    mean_average_precision = 0.0

    for query_id, query in enumerate(queries):
        relevant_docs = ground_truth.get(query_id, [])
        retrieved_documents = retrieve_documents(query, documents, word2vec_model, weights)
        avg_precision = calculate_average_precision(retrieved_documents, relevant_docs)
        mean_average_precision += avg_precision

    mean_average_precision /= len(ground_truth.keys())
    return mean_average_precision

# Load datasets and preprocess text
# documents_df = pd.read_csv('documents.csv')
documents_df = pd.read_csv('/home/kausar/Assignment1/documents.csv')
queries_df = pd.read_csv('/home/kausar/Assignment1/queries.csv')
ground_truth_df = pd.read_csv('/home/kausar/Assignment1/ground_truth.csv')


documents_df['processed_text'] = documents_df['text'].apply(preprocess_text)
queries_df['processed_text'] = queries_df['text'].apply(preprocess_text)

documents = documents_df['processed_text'].tolist()
queries = queries_df['processed_text'].tolist()
ground_truth = {}
for row in ground_truth_df.itertuples(index=False):
    query_id = row.query_id
    doc_id = row.doc_id

    if query_id not in ground_truth:
        ground_truth[query_id] = []

    ground_truth[query_id].append(doc_id)

# Download and load pre-trained FastText model "cc.en.300.bin"
# print("Downloading and loading the FastText model...")
# fasttext.util.download_model('en', if_exists='ignore')  # Download English model if not already downloaded
fasttext_model_path = r"/home/kausar/Assignment1/cc.en.300.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

# Convert FastText model to a format compatible with the `compute_embedding` function
class FastTextWrapper:
    def __init__(self, model):
        self.model = model
        self.vector_size = 300  # FastText English models have 300 dimensions

    def __contains__(self, token):
        return token in self.model

    def __getitem__(self, token):
        return self.model[token]

fasttext_wrapper = FastTextWrapper(fasttext_model)

# Define weights for scoring
weights = {'embedding': 1.0}

# Evaluate the retrieval system using only embeddings
map_score = evaluate_system(queries, ground_truth, documents, fasttext_wrapper, weights)
print(f"Mean Average Precision (MAP) for Embeddings: {map_score}")
