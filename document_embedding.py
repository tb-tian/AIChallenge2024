import faiss
import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import helpers
from helpers import get_logger
from load_all_video_keyframes_info import load_all_video_keyframes_info

all_video, video_keyframe_dict = load_all_video_keyframes_info()
logger = get_logger()


def embedding():
    documents = []
    embedding_info = []
    for v in all_video:
        doc_path = f"./data-staging/transcripts-en/{v}.txt"
        if not helpers.is_exits(doc_path):
            logger.warning(f"{doc_path} NOT FOUND!")
            raise Exception(f"missing {doc_path}")
            # continue
        # else:
        #     logger.debug(f"loading {doc_path}")
        with open(doc_path, "r") as file:
            document = file.read().splitlines()
            for i, line in enumerate(document):
                documents.append(line)
                embedding_info.append((v, i))

    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Normalize the document embeddings
    tfidf_matrix = normalize(tfidf_matrix, axis=1)

    # Create a FAISS index and add the document embeddings
    d = tfidf_matrix.shape[1]
    tfidf_index = faiss.IndexFlatL2(d)
    tfidf_index.add(tfidf_matrix.toarray().astype(np.float32))

    save_npz('./data-index/tfidf_matrix.npz', tfidf_matrix)
    joblib.dump(vectorizer, "./data-index/tfidf_vectorizer.pkl")
    joblib.dump(embedding_info, "./data-index/document_embedding_info.pkl")


def querying(query):
    # Load the TF-IDF matrix and the vectorizer
    tfidf_matrix = load_npz("./data-index/tfidf_matrix.npz")
    vectorizer = joblib.load("./data-index/tfidf_vectorizer.pkl")
    embedding_info = joblib.load("./data-index/document_embedding_info.pkl")

    # Vectorize the query
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity nbetween the query and the documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Find the indices of the top 5 most similar documents
    top_5_indices = cosine_similarities.argsort()[-5:][::-1]

    for i in top_5_indices:
        video = embedding_info[i][0]
        doc_path = f"./data-staging/transcripts-en/{video}.txt"
        chunk = embedding_info[i][1]
        print(video, chunk)
        with open(doc_path, "r") as file:
            lines = file.read().splitlines()
            chunk_line = lines[chunk]
            print(chunk_line)


if __name__ == "__main__":
    embedding()
    query = "A doctor/nurse is examining a patient's eyes using a machine called CLARUS. The patient wears a blue shirt. Ends with the scene of the doctor/nurse's hand adjusting the joystick with a glowing blue circle. What is the number recorded on the CLARUS machine?"
    querying(query)
