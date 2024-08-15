import streamlit as st
import numpy as np
import clip
import torch
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
st.set_page_config(layout="wide")
st.write("Hello")

keyframe_path = f"./datasets/keyframes/L01_V001"
keyframe_files = os.listdir(keyframe_path)
col1, col2, col3, col4 = st.columns(4)
WIDTH = 350
for i, file in enumerate(sorted(keyframe_files)):
    image_path = os.path.join(keyframe_path, file)
    image = Image.open(image_path)
    if i % 4 == 0:
        with col1:
            st.image(image, caption=f"{file}", width=WIDTH)
    elif i % 4 == 1:
        with col2:
            st.image(image, caption=f"{file}", width=WIDTH)
    elif i % 4 == 2:
        with col3:
            st.image(image, caption=f"{file}", width=WIDTH)
    else:
        with col4:
            st.image(image, caption=f"{file}", width=WIDTH)

# clip_path = np.load("./datasets/clip-features-vit-b32-sample/clip-features/L01_V001.npy")
# print(clip_path.shape)
