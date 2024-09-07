import time
from typing import Tuple

import faiss
import numpy as np
import open_clip

from hybrid_search import hibrid_search


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
        self.index = faiss.read_index("./data-index/embedding.index")
        self.embedding_info = np.load("./data-index/embedding_info.npy")

        print(f"loaded vectordb in {time.time()-start_time}s")

    def search_text(self, user_query, limit=100) -> Tuple:
        result = hibrid_search(user_query, limit)

        if not result:
            print("nothing found")
        return result
