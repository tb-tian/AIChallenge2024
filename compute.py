import streamlit as st
import numpy as np
import clip
import torch
from glob import glob
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

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


embedding_dict = {}
for v in all_video:
    clip_path = f'/home/thienan/Downloads/DataSampleAIC23-20240811T084355Z-002/DataSampleAIC23/clip-features-vit-b32-sample/clip-features/{v}.npy'
    a = np.load(clip_path)
    embedding_dict[v] = {}
    for i,k in enumerate(video_keyframe_dict[v]):
        embedding_dict[v][k] = a[i]
        



query = "Four people"
query = clip.tokenize(query).to(device)
query_feature = model.encode_text(query)



def cosine_similarity(query, data):
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2))
    return np.sum(data * query) / (query_norm * data_norm)


similarity_scores = []
for v in all_video:
    for k in video_keyframe_dict[v]:
        similarity = cosine_similarity(query_feature.detach().numpy(), embedding_dict[v][k])
        similarity_scores.append((v, k, similarity))

similarity_scores.sort(key=lambda x: x[2], reverse=True)
top_5 = similarity_scores[:5]


for v, k, similarity in top_5:
    file_path = f'/home/thienan/Downloads/DataSampleAIC23-20240811T084355Z-002/DataSampleAIC23/Keyframes_L01/keyframes/{v}/{k}.jpg'
    image = Image.open(file_path)
    image.show()
    print(f"Video: {v}, Keyframe: {k}, Similarity: {similarity}")