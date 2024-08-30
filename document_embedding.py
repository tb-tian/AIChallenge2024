import torch
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

def main():
    doc_path = "./datasets/texts/L01_V001_en.txt"
    with open(doc_path, 'r') as file:
        documents = file.read().splitlines()

    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    save_npz('tfidf_matrix.npz', tfidf_matrix)

    # Define the query and vectorize it
    query = "a World Cup cream cake"
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between the query and the documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Find the indices of the top 5 most similar documents
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]

    # Output the top 5 most similar documents
    top_5_documents = [documents[i] for i in top_5_indices]
    for idx, doc in enumerate(top_5_documents, 1):
        print(f"Top {idx} document: {doc}")

if __name__ == "__main__":
    main()