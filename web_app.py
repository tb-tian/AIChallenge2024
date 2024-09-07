import csv
import os
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image

from helpers import get_logger
from hybrid_search import hibrid_search, keyframe_querying
from vectordb import VectorDB

WIDTH = 350

if "cached" not in st.session_state:
    st.session_state["cached"] = {}
cached = st.session_state["cached"]

if "search_result" not in st.session_state:
    st.session_state["search_result"] = []
# search_result = st.session_state["search_result"]

logger = get_logger()


def get_kf_index(video, kf) -> Tuple[int, float]:  # frame_idx and start time
    transcript_path = f"./data-staging/audio-chunk-timestamps/{video}.csv"
    map_path = f"./data-source/map-keyframes/{video}.csv"
    with open(map_path) as map_file, open(transcript_path) as transcript_file:
        # lines = transcript_file.readlines()
        mapping = map_file.readlines()
        kf = int(kf)
        # start, end = lines[1:][kf - 1].split(",")
        n, pts_time, fps, frame_idx = mapping[kf].split(",")
    # start = int(start) / float(fps)
    # end = int(end) / float(fps)
    # print(start, end)
    return int(frame_idx), float(pts_time)


@st.dialog("Playing source video")
def play_dialog(img_source, video, kf):
    print(f"popup load ./data-source/videos/{video}.mp4", img_source, video, kf)
    frame_idx, pts_time = get_kf_index(video, kf)
    st.write(f"video: {video}, keyframe: {kf}, video start: {pts_time}")
    st.image(img_source)
    st.video(
        f"./data-source/videos/{video}.mp4",
        start_time=pts_time - 1,  # start 1 sec sooner
        autoplay=True,
    )
    print("POPup closed")


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
                # st.session_state["search_result"] = keyframe_querying(search_term)[:20]
                cached[search_term] = st.session_state["search_result"]

            os.makedirs("tmp/submission", exist_ok=True)
            outpath = f"tmp/submission/{query_id}.csv"
            is_qa = "qa" in query_id
            with open(outpath, "w") as f:
                exported_result = st.session_state["search_result"][:100]
                for vid, kf, sim in exported_result:
                    frame_idx, _ = get_kf_index(vid, kf)
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
        col_1, col_2, col_3, col_4 = st.columns(4)

        for i, (video, kf, similarity) in enumerate(st.session_state["search_result"]):
            similarity = round(similarity, 5)
            if i % 4 == 0:
                with col_1:
                    st.image(
                        f"./data-source/keyframes/{video}/{kf}.jpg",
                        caption=similarity,
                        width=WIDTH,
                    )
                    if st.button(f"view {video}/{kf}"):
                        play_dialog(
                            f"./data-source/keyframes/{video}/{kf}.jpg", video, kf
                        )
                    # st.checkbox(
                    #     "ok",
                    #     value=True,
                    #     key=f"{video}/{kf}",
                    #     label_visibility="hidden",
                    #     )
            elif i % 4 == 1:
                with col_2:
                    st.image(
                        f"./data-source/keyframes/{video}/{kf}.jpg",
                        caption=similarity,
                        width=WIDTH,
                    )
                    if st.button(f"view {video}/{kf}"):
                        play_dialog(
                            f"./data-source/keyframes/{video}/{kf}.jpg", video, kf
                        )
                    # st.checkbox(
                    #     "ok",
                    #     value=True,
                    #     key=f"{video}/{kf}",
                    #     label_visibility="hidden",
                    # )
            elif i % 4 == 2:
                with col_3:
                    st.image(
                        f"./data-source/keyframes/{video}/{kf}.jpg",
                        caption=similarity,
                        width=WIDTH,
                    )
                    if st.button(f"view {video}/{kf}"):
                        play_dialog(
                            f"./data-source/keyframes/{video}/{kf}.jpg", video, kf
                        )
                    # st.checkbox(
                    #     "ok",
                    #     value=True,
                    #     key=f"{video}/{kf}",
                    #     label_visibility="hidden",
                    # )
            else:
                with col_4:
                    st.image(
                        f"./data-source/keyframes/{video}/{kf}.jpg",
                        caption=similarity,
                        width=WIDTH,
                    )
                    if st.button(f"view {video}/{kf}"):
                        play_dialog(
                            f"./data-source/keyframes/{video}/{kf}.jpg", video, kf
                        )
                    # st.checkbox(
                    #     "ok",
                    #     value=True,
                    #     key=f"{video}/{kf}",
                    #     label_visibility="hidden",
                    # )
    else:
        st.text("search_result elem not available")
