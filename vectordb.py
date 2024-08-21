import time
from typing import Tuple
import numpy as np
import clip
import torch
import faiss


class VectorDB:
    def __init__(self):
        start_time = time.time()

        # clip model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, preprocess = clip.load("ViT-B/32", self.device)

        # Load the indexing database and embedding info
        self.index = faiss.read_index("./datasets/embedding.index")
        self.embedding_info = np.load("./datasets/info.npy")

        print(f"loaded vectordb in {time.time()-start_time}s")

    def search_text(self, user_query, limit=10) -> Tuple:
        query = clip.tokenize(user_query).to(self.device)
        query_feature = self.model.encode_text(query)

        # Query the faiss index to find the nearest neighbors
        query_embedding = (
            query_feature.detach().numpy().reshape(1, -1).astype("float32")
        )
        distances, indices = self.index.search(query_embedding, limit)

        # Retrieve the top 5 most suitable to the query
        top_5 = [
            (self.embedding_info[idx][0], self.embedding_info[idx][1], dist)
            for dist, idx in zip(distances[0], indices[0])
        ]

        # Print all the image that is suitable

        if not top_5:
            print("nothing found")
        return top_5
