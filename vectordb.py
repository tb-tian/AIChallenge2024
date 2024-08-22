import time
from typing import Tuple
import numpy as np
import open_clip
import faiss


class VectorDB:
    def __init__(self):
        start_time = time.time()

        # clip model setup
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Load the indexing database and embedding info
        self.index = faiss.read_index("./datasets/embedding.index")
        self.embedding_info = np.load("./datasets/info.npy")

        print(f"loaded vectordb in {time.time()-start_time}s")

    def search_text(self, user_query, limit=10) -> Tuple:
        query_feature = self.model.encode_text(self.tokenizer(user_query))

        # Query the faiss index to find the nearest neighbors
        query_embedding = (
            query_feature.detach().numpy().reshape(1, -1).astype("float32")
        )
        distances, indices = self.index.search(query_embedding, limit)

        # tuple of (video, keyframe, similarity)
        result = [
            (self.embedding_info[idx][0], self.embedding_info[idx][1], dist)
            for dist, idx in zip(distances[0], indices[0])
        ]
        if not result:
            print("nothing found")
        return result
