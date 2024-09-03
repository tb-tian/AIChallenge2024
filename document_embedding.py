import faiss
import joblib
import numpy as np
import torch
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from loading_dict import create_video_list_and_video_keyframe_dict

all_video, video_keyframe_dict = create_video_list_and_video_keyframe_dict()


def embedding():
    documents = []
    embedding_info = []
    for v in all_video:
        doc_path = f"./datasets/texts/{v}_en.txt"
        with open(doc_path, "r") as file:
            document = file.read().splitlines()
            for i, line in enumerate(document):
                documents.append(line)
                embedding_info.append((v, i))

    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Save the TF-IDF matrix
    save_npz("./datasets/tfidf_matrix.npz", tfidf_matrix)

    # Save the vectorizer
    joblib.dump(vectorizer, "./datasets/tfidf_vectorizer.pkl")

    # Save the embedding information
    joblib.dump(embedding_info, "./datasets/document_embedding_info.pkl")


def querying(query):
    # Load the TF-IDF matrix and the vectorizer
    tfidf_matrix = load_npz("./datasets/tfidf_matrix.npz")
    vectorizer = joblib.load("./datasets/tfidf_vectorizer.pkl")
    embedding_info = joblib.load("./datasets/document_embedding_info.pkl")

    # Vectorize the query
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between the query and the documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Find the indices of the top 5 most similar documents
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]

    for i in top_5_indices:
        video = embedding_info[i][0]
        doc_path = f"./datasets/texts/{video}_en.txt"
        chunk = embedding_info[i][1]
        print(video, chunk)
        with open(doc_path, "r") as file:
            lines = file.read().splitlines()
            chunk_line = lines[chunk]
            print(chunk_line)


if __name__ == "__main__":
    embedding()
    # query = "a World Cup cream cake"
    # querying(query)
