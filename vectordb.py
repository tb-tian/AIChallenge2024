import time
import streamlit as st
import numpy as np
import clip
import torch
from glob import glob
import faiss


class VectorDB:
    def __init__(self):
        start_time = time.time()
        # clip model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, preprocess = clip.load("ViT-B/32", self.device)

        # Create an array of keyframe and video. Remember to change the path that suitable to local machine
        all_keyframe = glob("./datasets/keyframes/*/*.jpg")
        video_keyframe_dict = {}
        all_video = glob("./datasets/keyframes/*")
        all_video = [v.rsplit("/", 1)[-1] for v in all_video]
        all_video = sorted(all_video)

        print(f"loaded {len(all_keyframe)} keyframes")
        print(f"loaded {len(all_video)} videos")

        for kf in all_keyframe:
            _, vid, kf = kf[:-4].rsplit("/", 2)
            if vid not in video_keyframe_dict.keys():
                video_keyframe_dict[vid] = [kf]
            else:
                video_keyframe_dict[vid].append(kf)

        for v, k in video_keyframe_dict.items():
            video_keyframe_dict[v] = sorted(k)

        # Load clip feature into an dictionary of numpy arrays
        embedding_dict = {}
        for v in all_video:
            clip_path = f"./datasets/clip-features-vit-b32-sample/clip-features/{v}.npy"
            a = np.load(clip_path)
            embedding_dict[v] = {}
            for i, k in enumerate(video_keyframe_dict[v]):
                embedding_dict[v][k] = a[i]

        # Flatten the embeddings and store them in a list along with their corresponding video and keyframe identifiers
        embedding_list = []
        self.embedding_info = []
        for v in all_video:
            for k in video_keyframe_dict[v]:
                embedding_list.append(embedding_dict[v][k])
                self.embedding_info.append((v, k))
        embedding_array = np.array(embedding_list)

        # Build the faiss index
        self.index = faiss.IndexFlatL2(embedding_array.shape[1])
        self.index.add(embedding_array)

        print(f"loaded vectordb in {time.time()-start_time}s")

    def search_text(self, user_query) -> str:
        query = clip.tokenize(user_query).to(self.device)
        query_feature = self.model.encode_text(query)

        # Query the faiss index to find the nearest neighbors
        query_embedding = (
            query_feature.detach().numpy().reshape(1, -1).astype("float32")
        )
        distances, indices = self.index.search(query_embedding, 5)

        # Retrieve the top 5 most suitable to the query
        top_5 = [
            (self.embedding_info[idx][0], self.embedding_info[idx][1], dist)
            for dist, idx in zip(distances[0], indices[0])
        ]

        # Print all the image that is suitable

        if not top_5:
            print("nothing found")
        return top_5
