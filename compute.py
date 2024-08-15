import streamlit as st
import numpy as np
import clip
import torch
from glob import glob
from PIL import Image

# clip model setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Create an array of keyframe and video. Remember to change the path that suitable to local machine
all_keyframe = glob('/home/thienan/Downloads/DataSampleAIC23-20240811T084355Z-002/DataSampleAIC23/Keyframes_L01/keyframes/*/*.jpg')
video_keyframe_dict = {}
all_video = glob('/home/thienan/Downloads/DataSampleAIC23-20240811T084355Z-002/DataSampleAIC23/Keyframes_L01/keyframes/*')
all_video = [v.rsplit('/',1)[-1] for v in all_video]
all_video = sorted(all_video)

for kf in all_keyframe:
    _, vid, kf = kf[:-4].rsplit('/',2)
    if vid not in video_keyframe_dict.keys():
        video_keyframe_dict[vid] = [kf]
    else:
        video_keyframe_dict[vid].append(kf)

for k,v in video_keyframe_dict.items():
    video_keyframe_dict[k] = sorted(v)

# Load clip feature into an dictionary of numpy arrays
embedding_dict = {}
for v in all_video:
    clip_path = f'/home/thienan/Downloads/DataSampleAIC23-20240811T084355Z-002/DataSampleAIC23/clip-features-vit-b32-sample/clip-features/{v}.npy'
    a = np.load(clip_path)
    embedding_dict[v] = {}
    for i,k in enumerate(video_keyframe_dict[v]):
        embedding_dict[v][k] = a[i]
        


# Query here
query = "England"
# Clip command to embedded the query
query = clip.tokenize(query).to(device) 
query_feature = model.encode_text(query)


# Find the similarity between 2 vectors
def cosine_similarity(query, data):
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2))
    return np.sum(data * query) / (query_norm * data_norm)

# Add all the similarity scores into an array
similarity_scores = []
for v in all_video:
    for k in video_keyframe_dict[v]:
        similarity = cosine_similarity(query_feature.detach().numpy(), embedding_dict[v][k])
        similarity_scores.append((v, k, similarity))

# Sort the array to get the top 5 most suitable to the query
similarity_scores.sort(key=lambda x: x[2], reverse=True)
top_5 = similarity_scores[:5]

# Print all the image that is suitable
for v, k, similarity in top_5:
    file_path = f'/home/thienan/Downloads/DataSampleAIC23-20240811T084355Z-002/DataSampleAIC23/Keyframes_L01/keyframes/{v}/{k}.jpg'
    image = Image.open(file_path)
    image.show()
    print(f"Video: {v}, Keyframe: {k}, Similarity: {similarity}")
    #aa