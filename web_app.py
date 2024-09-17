from copy import deepcopy
import csv
import os
from itertools import islice
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image

from helpers import get_logger
from hybrid_search import hybrid_search, keyframe_search
from videoqa import qa_engine
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
    map_path = f"./data-staging/map-keyframes/{video}.csv"
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
    st.markdown(
        """
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.header("TIAN™ Video Search")
    st.write(
        "Welcome to TIAN Video Search. You can blah blah blah here. And blah blah blah there also."
    )

    search_term = st.text_area("Search query (required):", height=100)
    qa_term = st.text_area("Question query (for qa query only):", height=25)
    col1, col2, col3 = st.columns(3)
    with col1:
        result_filename = st.text_input(
            "Result filename (must have one 'kis' or 'qa' in it)",
            value="query-0-kis.csv",
        )
    with col2:
        # keyframe_button = st.button("SEARCH Keyframe Only", type="primary")
        search_option = st.selectbox(
            "Choose your search type",
            ("keyframe_only", "hybrid"),
            index=0,
            placeholder="Choose your search type",
        )
    with col3:
        search_button = st.button("SEARCH!", type="primary")

    if "kis" in result_filename.lower():
        st.info(f"filename: {result_filename} => KIS query is using!", icon="ℹ️")
    elif "qa" in result_filename.lower():
        st.info(f"filename: {result_filename} => QA query is using!", icon="🔥")
        if qa_term == "":
            st.error("THIS IS QA query you should put something into qa_term => cancel querying")
            search_button = ""
    else:
        st.error(f"filename: {result_filename} => no KIS or QA is set cancel querying")
        search_button = ""
        

    if search_button:
        with st.spinner("Fetching Answer..."):
            search_term = search_term.strip()
            # play("video", "kf")
            logger.info("searching...", search_term)

            if cached.get(search_term):
                logger.info("fetch from cache")
                st.session_state["search_result"] = cached.get(search_term)
            else:
                logger.info("fetch from source")
                if search_option == "hybrid":
                    search_result = hybrid_search(search_term, limit=120)
                elif search_option == "keyframe_only":
                    search_result = keyframe_search(search_term, limit=120)
                else:
                    raise ValueError("invalid search_option")

                if "qa" in result_filename:
                    logger.info("this query is qa query, run qa model..")
                    new_search_result = []
                    for result in search_result:
                        file_path = (
                            f"./data-staging/keyframes/{result[0]}/{result[1]}.jpg"
                        )
                        response = qa_engine.ask(file_path, qa_term)
                        if len(response) > 0:
                            response = response[0]
                        else:
                            response = ""
                        new_search_result.append(result + (response,))
                else:
                    new_search_result = search_result

                st.session_state["search_result"] = new_search_result
                cached[search_term] = st.session_state["search_result"]

            os.makedirs("tmp/submission", exist_ok=True)
            outpath = f"tmp/submission/{result_filename}"
            is_qa = "qa" in result_filename
            with open(outpath, "w") as f:
                exported_result = st.session_state["search_result"][:100]
                for res in exported_result:
                    vid = res[0]
                    kf = res[1]
                    sim = res[2]
                    map_path = f"./data-staging/map-keyframes/{vid}.csv"
                    k = int(kf)
                    with open(map_path) as map_file:
                        map_file = csv.reader(map_file)
                        _, _, _, frame_idx = list(islice(map_file, k + 1))[k]
                    if is_qa:
                        f.write(f"{vid},{frame_idx},{res[3]}\n")
                    else:
                        f.write(f"{vid},{frame_idx}\n")

            st.write(f"exported to {outpath} with {len(exported_result)} results")
            logger.info(f"exported to {outpath}")
            download = st.download_button(
                f"Download {outpath}",
                data=open(outpath),
                file_name=f"{result_filename}.csv",
            )

    if st.session_state["search_result"]:
        col1, col2, col3, col4 = st.columns(4)

        for i, value in enumerate(st.session_state["search_result"]):
            video = value[0]
            kf = value[1]
            similarity = value[2:]
            # similarity = round(similarity, 5)
            file_path = f"./data-staging/keyframes/{video}/{kf}.jpg"
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
