import csv

import numpy as np
import streamlit as st
from PIL import Image

from vectordb import VectorDB

WIDTH = 350

if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = VectorDB()
vectordb = st.session_state["vectordb"]


@st.dialog("Playing source video")
def video_dialog(video, kf):
    txt_path = f"./datasets/preprocessing/{video}_scenes.txt"
    map_path = f"./datasets/map-keyframes/{video}.csv"
    with open(txt_path) as file, open(map_path) as map_file:
        lines = file.readlines()
        mapping = map_file.readlines()
        kf = int(kf)
        start, end = lines[kf - 1].split(" ")
        n, pts_time, fps, frame_idx = mapping[kf].split(",")
    start = int(start) / float(fps)
    end = int(end) / float(fps)
    print(start, end)
    st.write(f"{video}, {frame_idx}")
    st.video(
        f"./datasets/videos/{video}.mp4", start_time=start, end_time=end, autoplay=True
    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    hide_img_fs = """
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    </style>
    """

    st.markdown(hide_img_fs, unsafe_allow_html=True)

    st.header("TIAN™ Video Search")
    st.write(
        "Welcome to TIAN Video Search. You can blah blah blah here. And blah blah blah there also."
    )
    search_term = st.text_input("Search: ")

    print("searching...", search_term)

    res = vectordb.search_text(search_term)

    col1, col2, col3, col4 = st.columns(4)

    for i, (video, kf, similarity) in enumerate(res):
        file_path = f"./datasets/keyframes/{video}/{kf}.jpg"
        if i % 4 == 0:
            with col1:
                st.image(file_path, caption=similarity, width=WIDTH)
                if st.button(f"view {video}/{kf}"):
                    video_dialog(video, kf)
        elif i % 4 == 1:
            with col2:
                st.image(file_path, caption=similarity, width=WIDTH)
                if st.button(f"view {video}/{kf}"):
                    video_dialog(video, kf)

        elif i % 4 == 2:
            with col3:
                st.image(file_path, caption=similarity, width=WIDTH)
                if st.button(f"view {video}/{kf}"):
                    video_dialog(video, kf)

        else:
            with col4:
                st.image(file_path, caption=similarity, width=WIDTH)
                if st.button(f"view {video}/{kf}"):
                    video_dialog(video, kf)
