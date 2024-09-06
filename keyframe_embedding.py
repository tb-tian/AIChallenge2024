import time
from glob import glob

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

import helpers
from helpers import get_logger
from load_all_video_keyframes_info import load_all_video_keyframes_info

logger = get_logger()

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active


def embedding(pic_path):
    pic = Image.open(pic_path)
    pic = preprocess(pic).unsqueeze(0)
    with torch.no_grad():
        pic_feat = clip_model.encode_image(pic)
        pic_feat /= pic_feat.norm(dim=-1, keepdim=True)
    return pic_feat.cpu().numpy()


def main():
    all_video, video_keyframe_dict = load_all_video_keyframes_info()

    for v in tqdm(all_video, desc="Processing videos"):
        # Initialize an empty array with shape (0, 512)
        np_out = f"./data-staging/clip-features/{v}.npy"
        keyframe_array = np.empty((0, 512))
        if helpers.is_exits(np_out):
            logger.info(f"{np_out} exists, skip")
            continue
        for k in tqdm(
            video_keyframe_dict[v],
            desc=f"Processing keyframes for video {v}",
            leave=False,
        ):
            keyframe_path = f"./data-source/keyframes/{v}/{k}.jpg"
            keyframe_embedding = embedding(keyframe_path).reshape(1, -1)
            keyframe_array = np.vstack((keyframe_array, keyframe_embedding))
        np.save(np_out, keyframe_array)

    # Load clip feature into a dictionary of numpy arrays
    embedding_dict = {}
    for v in all_video:
        clip_path = f"./data-staging/clip-features/{v}.npy"
        a = np.load(clip_path)
        embedding_dict[v] = {}
        for i, k in enumerate(video_keyframe_dict[v]):
            embedding_dict[v][k] = a[i]

    # Save to embedding.index
    embedding_list = []
    embedding_info = []
    for v in all_video:
        for k in video_keyframe_dict[v]:
            embedding_list.append(embedding_dict[v][k])
            embedding_info.append((v, k))
    embedding_array = np.array(embedding_list)
    info_array = np.array(embedding_info)

    # Build the faiss index
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    faiss.write_index(index, "./data-index/embedding.index")

    # Save info_array into a npy file
    np.save("./data-index/embedding_info.npy", info_array)


if __name__ == "__main__":
    main()
