from itertools import islice
import streamlit as st
from hybrid_search import hybrid_search, keyframe_search
from helpers import get_logger
from search_ocr import search_by_ocr
from search_ocr import display_search_results
import csv
import os

WIDTH = 350
ocr_results_file = "./data-staging/ocr_results_en.csv"
map_keyframes_folder = "./data-staging/map-keyframes"

if "cached" not in st.session_state:
    st.session_state["cached"] = {}
cached = st.session_state["cached"]

if "search_result" not in st.session_state:
    st.session_state["search_result"] = []

if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = []

if "hybrid_results" not in st.session_state:
    st.session_state["hybrid_results"] = []

if "selected_keyframes" not in st.session_state:
    st.session_state["selected_keyframes"] = {}

logger = get_logger()


@st.dialog("Playing source video")
def play_dialog(video, kf, option):
    if option == "query":
        map_path = f"./data-staging/map-keyframes/{video}.csv"
        time_path = f"./data-staging/preprocessing/{video}_scenes.txt"
        with open(map_path) as map_file, open(time_path) as time_file:
            map_file = csv.reader(map_file)
            time_file = csv.reader(time_file, delimiter=" ")
            k = int(kf)
            _, _, fps, frame_idx = list(islice(map_file, k + 1))[k]
            start, end = list(islice(time_file, k))[k - 1]
    elif option == "ocr":
        map_path = f"./data-staging/map-keyframes/{video}.csv"
        time_path = f"./data-staging/preprocessing/{video}_scenes.txt"
        with open(map_path) as map_file, open(time_path) as time_file:
            map_file = csv.reader(map_file)
            time_file = csv.reader(time_file, delimiter=" ")
            k = int(kf)
            _, _, fps, frame_idx = list(islice(map_file, k + 1))[k]
            start, end = list(islice(time_file, k))[k - 1]
    st.write(f"{video},{frame_idx}")
    st.write(f"{fps}")
    st.video(
        f"./data-source/videos/{video}.mp4",
        autoplay=True,
        start_time=int(start) / int(fps[:-2]),
    )
    print("POPup closed")


@st.dialog("Zoom keyframe")
def zoom_image(file_path, video, kf, option):
    if option == "query":
        map_path = f"./data-staging/map-keyframes/{video}.csv"
        time_path = f"./data-staging/preprocessing/{video}_scenes.txt"
        with open(map_path) as map_file, open(time_path) as time_file:
            map_file = csv.reader(map_file)
            time_file = csv.reader(time_file, delimiter=" ")
            k = int(kf)
            _, _, _, frame_idx = list(islice(map_file, k + 1))[k]
    elif option == "ocr":
        map_path = f"./data-staging/map-keyframes/{video}.csv"
        time_path = f"./data-staging/preprocessing/{video}_scenes.txt"
        with open(map_path) as map_file, open(time_path) as time_file:
            map_file = csv.reader(map_file)
            time_file = csv.reader(time_file, delimiter=" ")
            k = int(kf)
            _, _, _, frame_idx = list(islice(map_file, k + 1))[k]
            print(kf)
            display_search_results(search_term, kf, video)

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

    search_option = st.radio("Search by:", ("Keyframe", "OCR", "Hybrid"))
    search_term = st.text_area("Ask a question here:", height=100)
    query_id = st.text_input(
        "Unique query id (used for export filename)", value="query-0-kis"
    )

    col1, col2 = st.columns(2)

    with col1:
        button = st.button("SEARCH", type="primary")
    with col2:
        export_selected_button = st.button("Export Selected Keyframes")

    download_placeholder = st.empty()

    if button:
        with st.spinner("Fetching Answer..."):
            st.session_state["ocr_results"] = []
            st.session_state["search_result"] = []
            st.session_state["hybrid_results"] = []
            st.session_state["selected_keyframes"] = {}
            search_term = search_term.strip()
            logger.info("searching...", search_term)

            if search_option == "Keyframe":
                if cached.get(search_term):
                    logger.info("fetch from cache")
                    st.session_state["search_result"] = cached.get(search_term)
                else:
                    logger.info("fetch from source")
                    st.session_state["search_result"] = keyframe_search(
                        search_term, limit=120
                    )
                    cached[search_term] = st.session_state["search_result"]
                search_results = st.session_state["search_result"]
            elif search_option == "OCR":
                st.session_state["ocr_results"] = search_by_ocr(search_term)
                search_results = st.session_state["ocr_results"]
            elif search_option == "Hybrid":
                if cached.get(search_term):
                    logger.info("fetch from cache")
                    st.session_state["hybrid_results"] = cached.get(search_term)
                else:
                    logger.info("fetch from source")
                    st.session_state["hybrid_results"] = hybrid_search(
                        search_term, limit=120
                    )
                    cached[search_term] = st.session_state["hybrid_results"]
                search_results = st.session_state["hybrid_results"]

    if (
        st.session_state["search_result"]
        or st.session_state["ocr_results"]
        or st.session_state["hybrid_results"]
    ):
        col1, col2, col3, col4 = st.columns(4)

        results = (
            st.session_state["search_result"]
            or st.session_state["ocr_results"]
            or st.session_state["hybrid_results"]
        )

        for i, result in enumerate(results):
            if (
                len(result) == 3
            ):  # For search_result and hybrid_results - keyframes and hybrid
                video, kf, similarity = result
                file_path = f"./data-staging/keyframes/{video}/{kf}.jpg"
                key = f"{video}/{kf}"
            else:  # For ocr_results
                subfolder, file_name, frame_idx, similarity = result
                file_path = f"./data-staging/keyframes/{subfolder}/{file_name}"
                key = f"{subfolder}/{file_name.split('.')[0]}"

            similarity = (
                round(float(similarity), 5)
                if isinstance(similarity, (int, float))
                else similarity
            )

            column = [col1, col2, col3, col4][i % 4]

            with column:
                st.image(file_path, caption=similarity, width=WIDTH)

                # Add checkbox
                if key not in st.session_state["selected_keyframes"]:
                    st.session_state["selected_keyframes"][key] = False
                st.session_state["selected_keyframes"][key] = st.checkbox(
                    f"Select {key}", key=f"checkbox_{key}"
                )

                button_col1, button_col2 = st.columns([1, 1])
                with button_col1:
                    if st.button(f"view {key}", key=f"view_{key}"):
                        play_dialog(
                            video if len(result) == 3 else subfolder,
                            kf if len(result) == 3 else file_name.split(".")[0],
                            "query" if len(result) == 3 else "ocr",
                        )
                with button_col2:
                    if st.button(f"zoom {key}", key=f"zoom_{key}"):
                        zoom_image(
                            file_path,
                            video if len(result) == 3 else subfolder,
                            kf if len(result) == 3 else file_name.split(".")[0],
                            "query" if len(result) == 3 else "ocr",
                        )

    if export_selected_button:
        results = (
            st.session_state["search_result"]
            or st.session_state["ocr_results"]
            or st.session_state["hybrid_results"]
        )

        selected_results = [
            result
            for result in results
            if (
                len(result) == 3
                and st.session_state["selected_keyframes"].get(
                    f"{result[0]}/{result[1]}", False
                )
            )
            or (
                len(result) == 4
                and st.session_state["selected_keyframes"].get(
                    f"{result[0]}/{result[1].split('.')[0]}", False
                )
            )
        ]

        unselected_results = [
            result
            for result in results
            if (
                len(result) == 3
                and not st.session_state["selected_keyframes"].get(
                    f"{result[0]}/{result[1]}", False
                )
            )
            or (
                len(result) == 4
                and not st.session_state["selected_keyframes"].get(
                    f"{result[0]}/{result[1].split('.')[0]}", False
                )
            )
        ]

        os.makedirs("tmp/submission", exist_ok=True)
        outpath = f"tmp/submission/{query_id}.csv"
        is_qa = "qa" in query_id

        with open(outpath, "w") as f:
            for result in selected_results:
                if (
                    len(result) == 3
                ):  # For search_result and hybrid_results - keyframes and hybrid
                    vid, kf, _ = result
                    map_path = f"./data-staging/map-keyframes/{vid}.csv"
                    k = int(kf)
                    with open(map_path) as map_file:
                        map_file = csv.reader(map_file)
                        _, _, _, frame_idx = list(islice(map_file, k + 1))[k]
                    if is_qa:
                        f.write(f"{vid},{frame_idx},\n")
                    else:
                        f.write(f"{vid},{frame_idx}\n")
                else:  # For ocr_results
                    subfolder, file_name, frame_idx, _ = result
                    f.write(f"{subfolder},{frame_idx}\n")
            for result in unselected_results:
                if (
                    len(result) == 3
                ):  # For search_result and hybrid_results - keyframes and hybrid
                    vid, kf, _ = result
                    map_path = f"./data-staging/map-keyframes/{vid}.csv"
                    k = int(kf)
                    with open(map_path) as map_file:
                        map_file = csv.reader(map_file)
                        _, _, _, frame_idx = list(islice(map_file, k + 1))[k]
                    if is_qa:
                        f.write(f"{vid},{frame_idx},\n")
                    else:
                        f.write(f"{vid},{frame_idx}\n")
                else:  # For ocr_results
                    subfolder, file_name, frame_idx, _ = result
                    f.write(f"{subfolder},{frame_idx}\n")

        logger.info(
            f"Exported {len(selected_results)} selected keyframes to the top of {outpath}"
        )
        with download_placeholder:
            if len(selected_results) > 0:
                logger.info(
                    f"Exported {len(selected_results)} selected keyframes to the top of {outpath}"
                )
                st.download_button(
                    f"Download submission file with {len(selected_results)} selected keyframes",
                    data=open(outpath),
                    file_name=f"{query_id}.csv",
                    key="download_selected",
                )
            else:
                logger.info(f"No selected keyframes to put on top of {outpath}")
                st.download_button(
                    f"Download submission file with original search results",
                    data=open(outpath),
                    file_name=f"{query_id}.csv",
                    key="download_no_selected",
                )
