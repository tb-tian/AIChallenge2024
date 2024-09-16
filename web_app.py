import csv
import os
from itertools import islice
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image

from helpers import get_logger
from hybrid_search import hibrid_search, keyframe_querying
from vectordb import VectorDB
from helpers import get_logger

WIDTH = 350

if "cached" not in st.session_state:
    st.session_state["cached"] = {}
cached = st.session_state["cached"]

if "search_result" not in st.session_state:
    st.session_state["search_result"] = []

logger = get_logger()


@st.dialog("Playing source video")
def play_dialog(video, kf):
    map_path = f"./data-source/map-keyframes/{video}.csv"
    time_path = f"./data-staging/preprocessing/{video}_scenes.txt"
    with open(map_path) as map_file, open(time_path) as time_file:
        map_file = csv.reader(map_file)
        time_file = csv.reader(time_file, delimiter=" ")
        k = int(kf)
        n, pts_time, fps, frame_idx = list(islice(map_file, k + 1))[k]
        start, end = list(islice(time_file, k))[k - 1]
    st.write(f"{video},{frame_idx}")
    st.video(
        f"./data-source/videos/{video}.mp4",
        autoplay=True,
        start_time=int(start) / int(fps[:-2]),
        end_time=int(end) / int(fps[:-2]),
    )
    print("POPup closed")


@st.dialog("Zoom keyframe")
def zoom_image(file_path, video, kf):
    map_path = f"./data-staging/map-keyframes/{video}.csv"
    time_path = f"./data-staging/preprocessing/{video}_scenes.txt"
    with open(map_path) as map_file, open(time_path) as time_file:
        map_file = csv.reader(map_file)
        time_file = csv.reader(time_file, delimiter=" ")
        k = int(kf)
        n, pts_time, fps, frame_idx = list(islice(map_file, k + 1))[k]
        start, end = list(islice(time_file, k))[k - 1]

    st.image(file_path, caption=f"{video},{frame_idx}", use_column_width=True)


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

    search_term = st.text_area("Ask a question here:", height=100)
    query_id = st.text_input(
        "Unique query id (used for export filename)", value="query-0-kis"
    )
    button = st.button("SEARCH", type="primary")

    if button:
        with st.spinner("Fetching Answer..."):
            search_term = search_term.strip()
            # play("video", "kf")
            logger.info("searching...", search_term)

            if cached.get(search_term):
                logger.info("fetch from cache")
                st.session_state["search_result"] = cached.get(search_term)
            else:
                logger.info("fetch from source")
                st.session_state["search_result"] = hibrid_search(
                    search_term, limit=120
                )
                # search_result = keyframe_querying(search_term)[:20]
                cached[search_term] = st.session_state["search_result"]

            os.makedirs("tmp/submission", exist_ok=True)
            outpath = f"tmp/submission/{query_id}.csv"
            is_qa = "qa" in query_id
            with open(outpath, "w") as f:
                exported_result = st.session_state["search_result"][:100]
                for vid, kf, sim in exported_result:
                    map_path = f"./data-source/map-keyframes/{vid}.csv"
                    k = int(kf)
                    with open(map_path) as map_file:
                        map_file = csv.reader(map_file)
                        _, _, _, frame_idx = list(islice(map_file, k + 1))[k]
                    if is_qa:
                        f.write(f"{vid},{frame_idx},\n")
                    else:
                        f.write(f"{vid},{frame_idx}\n")

            st.write(f"exported to {outpath} with {len(exported_result)} results")
            logger.info(f"exported to {outpath}")
            download = st.download_button(
                f"Download {outpath}", data=open(outpath), file_name=f"{query_id}.csv"
            )

    if st.session_state["search_result"]:
        col1, col2, col3, col4 = st.columns(4)

        for i, (video, kf, similarity) in enumerate(st.session_state["search_result"]):
            similarity = round(similarity, 5)
            file_path = f"./data-source/keyframes/{video}/{kf}.jpg"
            if i % 4 == 0:
                with col1:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf)
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf)

            elif i % 4 == 1:
                with col2:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf)
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf)

            elif i % 4 == 2:
                with col3:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf)
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf)

            else:
                with col4:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf)
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf)
