import pandas as pd
import numpy as np
import fasttext
import fasttext.util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import string
import os

# Download necessary resources for nltk
download('punkt')
download('stopwords')

# Define file paths
documents_path = '/home/kausar/Assignment1/documents.csv'
queries_path = '/home/kausar/Assignment1/queries.csv'
ground_truth_path = '/home/kausar/Assignment1/ground_truth.csv'
pretrained_model_path = r"/home/kausar/Assignment1/cc.en.300.bin"
pretrained_vectors_path = r"/home/kausar/Assignment1/cc.en.300.vec"
fine_tuned_model_path = "fine_tuned_cisi_fasttext.bin"
combined_text_path = "cisi_combined_text.txt"

# Step 1: Convert .bin to .vec if necessary
if not os.path.exists(pretrained_vectors_path):
    print("Converting .bin to .vec format...")
    os.system(f"fasttext print-word-vectors {pretrained_model_path} > {pretrained_vectors_path}")

# Step 2: Load pre-trained FastText model and print dimensions
print("Loading pre-trained FastText model...")
pretrained_model = fasttext.load_model(pretrained_model_path)
pretrained_dimension = pretrained_model.get_dimension()
print(f"Pre-trained model dimensions: {pretrained_dimension}")

# Step 3: Load datasets
documents_df = pd.read_csv(documents_path)
queries_df = pd.read_csv(queries_path)
ground_truth_df = pd.read_csv(ground_truth_path)

# Step 4: Define stopwords and preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.notna(text):
        tokens = word_tokenize(str(text).lower())
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    else:
        return ''

# Step 5: Preprocess the text in documents and queries
documents_df['processed_text'] = documents_df['text'].apply(preprocess_text)
queries_df['processed_text'] = queries_df['text'].apply(preprocess_text)

# Step 6: Combine preprocessed text into a single file for fine-tuning
with open(combined_text_path, 'w') as f:
    for text in documents_df['processed_text']:
        f.write(text + '\n')
    for text in queries_df['processed_text']:
        f.write(text + '\n')

# Step 7: Fine-tune FastText model
print("Fine-tuning FastText model...")
fine_tuned_model = fasttext.train_unsupervised(
    input=combined_text_path,
    model='skipgram',  # or 'cbow'
    dim=pretrained_dimension,  # Match pre-trained model's dimension dynamically
    epoch=5,
    lr=0.05,
    pretrainedVectors=pretrained_vectors_path
)

# Step 8: Save the fine-tuned model
fine_tuned_model.save_model(fine_tuned_model_path)
print(f"Fine-tuned FastText model saved to {fine_tuned_model_path}")

# Step 9: Function to compute embeddings using the fine-tuned model
def compute_embedding(text, model):
    return model.get_sentence_vector(text)

# Step 10: Create ground truth mapping
ground_truth = {}
for row in ground_truth_df.itertuples(index=False):
    query_id = row.query_id
    doc_id = row.doc_id
    if query_id not in ground_truth:
        ground_truth[query_id] = []
    ground_truth[query_id].append(doc_id)

# Step 11: Function to retrieve top 10 documents based on fine-tuned FastText embeddings
def retrieve_top_10_fasttext(query_text, documents_df, model):
    query_embedding = compute_embedding(query_text, model)
    similarities = []
    for doc_id, doc_text in zip(documents_df['doc_id'], documents_df['processed_text']):
        doc_embedding = compute_embedding(doc_text, model)
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        similarities.append((doc_id, similarity))
    ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    return ranked_docs

# Step 12: Function to calculate Average Precision (AP) for a single query
def calculate_average_precision(ranked_docs, relevant_docs):
    precision_at_k = []
    num_relevant_docs = 0
    for k, (doc_id, _) in enumerate(ranked_docs):
        if doc_id in relevant_docs:
            num_relevant_docs += 1
            precision_at_k.append(num_relevant_docs / (k + 1))
    return np.mean(precision_at_k) if precision_at_k else 0.0

# Step 13: Function to evaluate MAP for all queries
def evaluate_map(queries_df, ground_truth, documents_df, model):
    mean_average_precision = 0.0
    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        query_text = row['processed_text']
        relevant_docs = ground_truth.get(query_id, [])
        ranked_docs = retrieve_top_10_fasttext(query_text, documents_df, model)
        average_precision = calculate_average_precision(ranked_docs, relevant_docs)
        mean_average_precision += average_precision
    return mean_average_precision / len(ground_truth.keys())

# Step 14: Evaluate MAP for the fine-tuned FastText model
map_score = evaluate_map(queries_df, ground_truth, documents_df, fine_tuned_model)
print(f"Mean Average Precision (MAP) for Fine-Tuned FastText Model on CISI Dataset: {map_score:.4f}")
