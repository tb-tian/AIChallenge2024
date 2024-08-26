import torch
import open_clip
import faiss
import numpy as np

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def embedding(text):
    text_feat = model.encode_text(tokenizer(text))
    text_embedding = text_feat.detach().numpy().reshape(1,-1).astype("float32")
    return text_embedding

doc_path = "./datasets/texts/L01_V001_en.txt"
embedding_list = []
line_list = []

with open(doc_path, 'r') as file:
    for i, line in enumerate(file):
        line_list.append(line)
        embedding_list.append(embedding(line))

embedding_array = np.concatenate(embedding_list, axis=0)

query = "Art exhibition"
query_feature = model.encode_text(tokenizer(query))

# Query the faiss index to find the nearest neighbors
query_embedding = (
    query_feature.detach().numpy().reshape(1, -1).astype("float32")
)

# Build the faiss index
index = faiss.IndexFlatL2(embedding_array.shape[1])
index.add(embedding_array)

distances, indices = index.search(query_embedding, 5)

for dist, idx in zip(distances[0], indices[0]):
    print(line_list[idx], dist)