import csv

import faiss
import joblib
import numpy as np
import open_clip
import pandas as pd
from PIL import Image
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm

from helpers import get_logger
from load_all_video_keyframes_info import load_all_video_keyframes_info

all_video, video_keyframe_dict = load_all_video_keyframes_info()
logger = get_logger()


def keyframe_querying(query):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Load the keyframe embedding from the FAISS index
    keyframe_index = faiss.read_index("./data-index/embedding.index")

    # Extract the vectors from the FAISS index
    keyframe_embeddings = keyframe_index.reconstruct_n(0, keyframe_index.ntotal)
    keyframe_embeddings = np.array(keyframe_embeddings)
    embedding_info = np.load("./data-index/embedding_info.npy")

    print(f"loaded {int(keyframe_embeddings.size / 512)} keyframes")

    keyframe_embeddings = normalize(keyframe_embeddings, axis=1)

    # Embedding and query the faiss index to find the nearest keyframe
    query_feature = model.encode_text(tokenizer(query))
    query_embedding = query_feature.detach().numpy().reshape(1, -1).astype("float32")
    query_embedding = normalize(query_embedding, axis=1)
    limit = keyframe_index.ntotal
    distances, indices = keyframe_index.search(query_embedding, limit)
    similarity_scores = 1 / (distances + 1e-8)
    distance_array = [
        (embedding_info[idx][0], embedding_info[idx][1], dist)
        for dist, idx in zip(similarity_scores[0], indices[0])
    ]

    result = {}
    for video, kf, dist in distance_array:
        if video not in result:
            result[video] = {}
        result[video][kf] = dist
    return result


def document_querying(query):
    # Load the data
    document_index = faiss.read_index("./data-index/tfidf.index")
    vectorizer = joblib.load("./data-index/tfidf_vectorizer.pkl")
    embedding_info = joblib.load("./data-index/document_embedding_info.pkl")
    mapping_df = pd.read_csv("./data-index/mapping.csv", dtype={"keyframe": str})

    print(f"loaded {document_index.ntotal} documents")

    query_feature = vectorizer.transform([query])
    query_embedding = query_feature.toarray().reshape(1, -1).astype("float32")
    query_embedding = normalize(query_embedding, axis=1)

    # Search the FAISS index
    limit = document_index.ntotal
    distances, indices = document_index.search(query_embedding, limit)

    result = {}
    for idx, score in zip(indices[0], distances[0]):
        video, chunk = embedding_info[idx]

        mapping_row = mapping_df[
            (mapping_df["video"] == video) & (mapping_df["chunk"] == chunk)
        ]
        for kf in mapping_row["keyframe"]:
            if video not in result:
                result[video] = {}
            result[video][kf] = score

    return result


def sort_results(result):
    # Flatten the nested dictionary into a list of tuples
    flattened_results = [
        (video, kf, score) for video in result for kf, score in result[video].items()
    ]
    # Sort the list of tuples based on the score in descending order
    sorted_results = sorted(flattened_results, key=lambda x: x[2], reverse=True)
    # Include rank in the sorted results
    ranked_results = [
        (rank + 1, video, kf, score)
        for rank, (video, kf, score) in enumerate(sorted_results)
    ]
    return ranked_results

def keyframe_search(query, limit=100):
    logger.debug(f"keyframe_querying: {query}")
    kf_res = keyframe_querying(query)

    ranked_kf_res = sort_results(kf_res)

    ranked_kf_dic = {}

    for rank, video, kf, score in ranked_kf_res:
        if video not in ranked_kf_dic:
            ranked_kf_dic[video] = {}
        ranked_kf_dic[video][kf] = rank
    
    logger.debug("rerank")
    rerank = []
    for v in all_video:
        for kf in video_keyframe_dict[v]:
            if ranked_kf_dic[v].get(kf) is None:
                continue
            rerank.append(
                (
                    v,
                    kf,
                    1 / ranked_kf_dic[v].get(kf, 0)
                )
            )
            # rerank.append((v, kf, kf_res[v][kf]))

    rerank = sorted(rerank, key=lambda x: x[2], reverse=True)
    rerank = rerank[:limit]

    return rerank

def hibrid_search(query, limit=100):
    logger.debug(f"keyframe_querying: {query}")
    kf_res = keyframe_querying(query)
    logger.debug(f"document_querying: {query}")
    doc_res = document_querying(query)

    ranked_kf_res = sort_results(kf_res)
    ranked_doc_res = sort_results(doc_res)

    ranked_kf_dic = {}
    # print("Sorted Keyframe Results:")
    for rank, video, kf, score in ranked_kf_res:
        if video not in ranked_kf_dic:
            ranked_kf_dic[video] = {}
        ranked_kf_dic[video][kf] = rank

    ranked_doc_dic = {}
    # # # print("\nSorted Document Results:")
    for rank, video, kf, score in ranked_doc_res:
        if video not in ranked_doc_dic:
            ranked_doc_dic[video] = {}
        ranked_doc_dic[video][kf] = rank
        # print(f"Rank: {rank}, Video: {video}, Keyframe: {kf}, Score: {score}")

    # for v in all_video:
    #     for kf in video_keyframe_dict[v]:
    #         print(ranked_kf_dic[v][kf], ranked_doc_dic[v][kf])

    logger.debug("rerank")
    rerank = []
    for v in all_video:
        for kf in video_keyframe_dict[v]:
            if ranked_kf_dic[v].get(kf) is None:
                continue
            if ranked_doc_dic[v].get(kf) is None:
                continue
            rerank.append(
                (
                    v,
                    kf,
                    0.7 / ranked_kf_dic[v].get(kf, 0)
                    + 0.3 / ranked_doc_dic[v].get(kf, 0),
                )
            )
            # rerank.append((v, kf, kf_res[v][kf]))

    rerank = sorted(rerank, key=lambda x: x[2], reverse=True)
    rerank = rerank[:limit]

    return rerank


if __name__ == "__main__":
    res = hibrid_search("car", 20)
    print(res)

    for video, keyframe, similarity in res:
        file_path = f"./data-source/keyframes/{video}/{keyframe}.jpg"
        image = Image.open(file_path)
        image.show()
        print(f"Video: {video}, Keyframe: {keyframe}, Similarity: {similarity}")
