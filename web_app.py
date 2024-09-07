import csv

import numpy as np
import streamlit as st
from PIL import Image

from hybrid_search import hibrid_search, keyframe_querying
from vectordb import VectorDB
from helpers import get_logger
WIDTH = 350

if "cached" not in st.session_state:
    st.session_state["cached"] = {}
cached = st.session_state["cached"]

if "search_result" not in st.session_state:
    st.session_state["search_result"] = []
search_result = st.session_state["search_result"]

logger = get_logger()

@st.dialog("Playing source video")
def play_dialog(video, kf):
    st.write(f"{video} - {kf}")

    # txt_path = f"./data-staging/transcripts/{video}.txt"
    # map_path = f"./data-source/map-keyframes/{video}.csv"
    # with open(txt_path) as file, open(map_path) as map_file:
    #     lines = file.readlines()
    #     mapping = map_file.readlines()
    #     kf = int(kf)
    #     start, end = lines[kf - 1].split(" ")
    #     n, pts_time, fps, frame_idx = mapping[kf].split(",")
    # start = int(start) / float(fps)
    # end = int(end) / float(fps)
    # print(start, end)
    # st.write(f"{video}, {frame_idx}")
    st.video(
        f"./data-source/videos/{video}.mp4", autoplay=True,
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

    with st.container():
        search_term = st.text_area("Ask a question here:", height=100, key="query_text")
        button = st.button("Submit", key="button")
        if button:
            with st.spinner("Fetching Answer..."):
                search_term = search_term.strip()
                # play("video", "kf")
                logger.info("searching...", search_term)

                if cached.get(search_term):
                    logger.info("fetch from cache")
                    search_result = cached.get(search_term)
                else:
                    logger.info("fetch from source")
                    search_result = hibrid_search(search_term)
                    # search_result = keyframe_querying(search_term)[:20]
                    cached[search_term] = search_result


        if search_result:
            col1, col2, col3, col4 = st.columns(4)

            for i, (video, kf, similarity) in enumerate(search_result):
                similarity = round(similarity, 5)
                file_path = f"./data-source/keyframes/{video}/{kf}.jpg"
                if i % 4 == 0:
                    with col1:
                        st.image(file_path, caption=similarity, width=WIDTH)
                        st.button(f"view {video}/{kf}", on_click=play_dialog, args=[video, kf])
                       
                elif i % 4 == 1:
                    with col2:
                        st.image(file_path, caption=similarity, width=WIDTH)
                        st.button(f"view {video}/{kf}", on_click=play_dialog, args=[video, kf])

                elif i % 4 == 2:
                    with col3:
                        st.image(file_path, caption=similarity, width=WIDTH)
                        st.button(f"view {video}/{kf}", on_click=play_dialog, args=[video, kf])

                else:
                    with col4:
                        st.image(file_path, caption=similarity, width=WIDTH)
                        st.button(f"view {video}/{kf}", on_click=play_dialog, args=[video, kf])
