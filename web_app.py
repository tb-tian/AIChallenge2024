import streamlit as st
import numpy as np
from PIL import Image

from vectordb import VectorDB

WIDTH = 350

if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = VectorDB()
vectordb = st.session_state["vectordb"]


@st.dialog("Playing source video")
def video_dialog(source):
    st.write(f"Source {source}")
    st.video(f"./datasets/video/{source}.mp4")


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
                    video_dialog(video)
        elif i % 4 == 1:
            with col2:
                st.image(file_path, caption=similarity, width=WIDTH)
                if st.button(f"view {video}/{kf}"):
                    video_dialog(video)

        elif i % 4 == 2:
            with col3:
                st.image(file_path, caption=similarity, width=WIDTH)
                if st.button(f"view {video}/{kf}"):
                    video_dialog(video)

        else:
            with col4:
                st.image(file_path, caption=similarity, width=WIDTH)
                if st.button(f"view {video}/{kf}"):
                    video_dialog(video)
