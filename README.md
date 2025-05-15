# FastText-Enhanced-IR-System

This repository contains Python code for an Information Retrieval (IR) system that leverages embeddings, specifically Fasttext, in combination with traditional techniques like TF-IDF and BM25 for document retrieval. The system aims to improve the accuracy of document ranking by integrating both modern word embedding methods and conventional IR approaches.

## Overview

This IR system is implemented in Python and consists of several components for document retrieval and evaluation. The code includes preprocessing of text data, retrieval methods combining TF-IDF, BM25, and Word2Vec embeddings, as well as evaluation metrics for system performance.

## Project Structure

- `coding/`: Contains Python scripts for the IR system along with dataset
- `Readme.md`: Overview of the project and instructions for setup.

## System Requirements
Python 3.6+


## Setup Instructions

1. **Installation**: Ensure all necessary Python packages are installed:


2. **Dataset Setup**: Place the documents, queries, and ground truth relevance judgments in the same directory.

3. **Running the Code**: 
Execute for example the script `Isolated_Fasttext.py` to perform the retrieval process for pretrained model of Fasttext. 
Execute the script `Training_Fasttext.py` to perform the retrieval process for model which is being trained. 
Execute the script `Fine-tune_Fasttext.py` to perform the retrieval process for fine tuning fasttext modeol.
For Isolated results execute the script `BM25_isolated.py`, TFIDF_isolated.py
## Components

### Preprocessing
- `preprocess_text`: Function for text normalization, including tokenization, lowercase conversion, and removal of punctuation and stopwords.

### Retrieval System
- `retrieve_documents`: Retrieves and ranks relevant documents using TF-IDF, BM25, and Fasttext embeddings with weighted combination scores.

### Evaluation Metrics
- Evaluation functions calculate precision, recall, and average precision, culminating in Mean Average Precision (MAP) for the retrieval system.

## Usage

- Adjust parameters, weights, or configurations in the code to experiment with different settings.
- Evaluation metrics and results will be printed to the console.

## Files
- `documents.csv`: Dataset containing documents for the IR system.
- `queries.csv`: Dataset comprising user queries.
- `ground_truth.csv`: Dataset with ground truth relevance judgments.

##Results
After running the code, the console will display the calculated Mean Average Precision (MAP) score for the implemented retrieval system using the provided datasets and configurations.


