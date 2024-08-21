import torch
from PIL import Image
import numpy as np
import open_clip
from glob import glob
import time
import faiss
from tqdm import tqdm

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active


def embedding(pic_path):
    pic = Image.open(pic_path)
    pic = preprocess(pic).unsqueeze(0)
    with torch.no_grad():
        pic_feat = model.encode_image(pic)
        pic_feat /= pic_feat.norm(dim=-1, keepdim=True)
    return pic_feat.cpu().numpy()


def main():
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

    for k, v in video_keyframe_dict.items():
        video_keyframe_dict[k] = sorted(v)

    for v in tqdm(all_video, desc="Processing videos"):
        keyframe_array = np.empty(
            (0, 512)
        )  # Initialize an empty array with shape (0, 512)
        for k in tqdm(
            video_keyframe_dict[v],
            desc=f"Processing keyframes for video {v}",
            leave=False,
        ):
            keyframe_path = f"./datasets/keyframes/{v}/{k}.jpg"
            keyframe_embedding = embedding(keyframe_path).reshape(1, -1)
            keyframe_array = np.vstack((keyframe_array, keyframe_embedding))
        np.save(f"./datasets/clip-features/{v}.npy", keyframe_array)

    # Load clip feature into an dictionary of numpy arrays
    embedding_dict = {}
    for v in all_video:
        clip_path = f"./datasets/clip-features/{v}.npy"
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
    faiss.write_index(index, "./datasets/embedding.index")

    # Save info_array into a npy file
    np.save("./datasets/info.npy", info_array)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Embedding time: ", end - start)
