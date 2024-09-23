import csv
import os
from itertools import islice
import streamlit as st
from hybrid_search import hybrid_search
from helpers import get_logger

WIDTH = 350
ocr_results_file = "data-staging/ocr_results_en.csv"
map_keyframes_folder = "data-staging/map-keyframes"

if "cached" not in st.session_state:
    st.session_state["cached"] = {}
cached = st.session_state["cached"]

if "search_result" not in st.session_state:
    st.session_state["search_result"] = []

if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = []

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
        frame_idx = kf
        fps = "25.0"
        start = str(int(frame_idx) - 50)
        st.write(f"{fps}")
    st.write(f"{video},{frame_idx}")
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
        frame_idx = kf
    st.image(file_path, caption=f"{video},{frame_idx}", use_column_width=True)


def load_csv_file(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: {file_path} does not exist.")
        return []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def find_match_in_ocr(input_text, ocr_data):
    results = []
    for row in ocr_data:
        if input_text.lower() in row["text"].lower():
            results.append(row)
    return results


def find_frame_idx(subfolder, file_name):
    map_keyframes_file = os.path.join(map_keyframes_folder, f"{subfolder}.csv")
    keyframes_data = load_csv_file(map_keyframes_file)

    if not keyframes_data:
        return None

    file_number = int(file_name.split(".")[0])
    file_number_adjusted = file_number + 1

    for row in keyframes_data:
        if int(row["n"]) == file_number_adjusted:
            return row["frame_idx"]
    return None


def search_by_ocr(input_text):
    ocr_data = load_csv_file(ocr_results_file)
    if not ocr_data:
        st.error("No OCR data found.")
        return []

    matches = find_match_in_ocr(input_text, ocr_data)
    if not matches:
        reversed_text = input_text[::-1]
        matches = find_match_in_ocr(reversed_text, ocr_data)
        if not matches:
            st.info("No OCR matches found.")
            return []

    result_data = []
    for match in matches:
        subfolder = match["subfolder"]
        file_name = match["file_name"]
        frame_idx = find_frame_idx(subfolder, file_name)
        if frame_idx:
            result_data.append((subfolder, file_name, frame_idx, 0.0))
    return result_data


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

    search_option = st.radio("Search by:", ("Query", "OCR"))
    search_term = st.text_area("Ask a question here:", height=100)
    query_id = st.text_input(
        "Unique query id (used for export filename)", value="query-0-kis"
    )
    button = st.button("SEARCH", type="primary")

    if button:
        with st.spinner("Fetching Answer..."):
            st.session_state["ocr_results"] = []
            st.session_state["search_result"] = []
            search_term = search_term.strip()
            logger.info("searching...", search_term)

            if search_option == "Query":
                if cached.get(search_term):
                    logger.info("fetch from cache")
                    st.session_state["search_result"] = cached.get(search_term)
                else:
                    logger.info("fetch from source")
                    st.session_state["search_result"] = hybrid_search(
                        search_term, limit=120
                    )
                    cached[search_term] = st.session_state["search_result"]
                search_results = st.session_state["search_result"]
            elif search_option == "OCR":
                st.session_state["ocr_results"] = search_by_ocr(search_term)

            os.makedirs("tmp/submission", exist_ok=True)
            outpath = f"tmp/submission/{query_id}.csv"
            is_qa = "qa" in query_id
            with open(outpath, "w") as f:
                if search_option == "Query":
                    exported_result = st.session_state["search_result"][:100]
                elif search_option == "OCR":
                    exported_result = st.session_state["ocr_results"][:100]

                if search_option == "Query":
                    for vid, kf, sim in exported_result:
                        map_path = f"./data-staging/map-keyframes/{vid}.csv"
                        k = int(kf)
                        with open(map_path) as map_file:
                            map_file = csv.reader(map_file)
                            _, _, _, frame_idx = list(islice(map_file, k + 1))[k]
                        if is_qa:
                            f.write(f"{vid},{frame_idx},\n")
                        else:
                            f.write(f"{vid},{frame_idx}\n")
                elif search_option == "OCR":
                    for subfolder, file_name, frame_idx, _ in exported_result:
                        f.write(f"{subfolder},{frame_idx}\n")

            st.write(f"exported to {outpath} with {len(exported_result)} results")
            logger.info(f"exported to {outpath}")
            download = st.download_button(
                f"Download {outpath}", data=open(outpath), file_name=f"{query_id}.csv"
            )

    if st.session_state["search_result"]:
        col1, col2, col3, col4 = st.columns(4)

        for i, (video, kf, similarity) in enumerate(st.session_state["search_result"]):
            similarity = round(similarity, 5)
            file_path = f"./data-staging/keyframes/{video}/{kf}.jpg"
            if i % 4 == 0:
                with col1:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf, "query")
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf, "query")

            elif i % 4 == 1:
                with col2:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf, "query")
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf, "query")

            elif i % 4 == 2:
                with col3:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf, "query")
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf, "query")

            else:
                with col4:
                    st.image(file_path, caption=similarity, width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"view {video}/{kf}"):
                            play_dialog(video, kf, "query")
                    with button_col2:
                        if st.button(f"zoom {video}/{kf}"):
                            zoom_image(file_path, video, kf, "query")

    elif st.session_state["ocr_results"]:
        col1, col2, col3, col4 = st.columns(4)

        for i, (subfolder, file_name, frame_idx, similarity) in enumerate(
            st.session_state["ocr_results"]
        ):
            file_path = f"./data-staging/keyframes/{subfolder}/{file_name}"

            if i % 4 == 0:
                with col1:
                    st.image(file_path, caption="OCR Result", width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"View {subfolder}/{frame_idx}"):
                            play_dialog(subfolder, frame_idx, "ocr")
                    with button_col2:
                        if st.button(f"Zoom {subfolder}/{frame_idx}"):
                            zoom_image(file_path, subfolder, frame_idx, "ocr")
            elif i % 4 == 1:
                with col2:
                    st.image(file_path, caption="OCR Result", width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"View {subfolder}/{frame_idx}"):
                            play_dialog(subfolder, frame_idx, "ocr")
                    with button_col2:
                        if st.button(f"Zoom {subfolder}/{frame_idx}"):
                            zoom_image(file_path, subfolder, frame_idx, "ocr")
            elif i % 4 == 2:
                with col3:
                    st.image(file_path, caption="OCR Result", width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"View {subfolder}/{frame_idx}"):
                            play_dialog(subfolder, frame_idx, "ocr")
                    with button_col2:
                        if st.button(f"Zoom {subfolder}/{frame_idx}"):
                            zoom_image(file_path, subfolder, frame_idx, "ocr")
            else:
                with col4:
                    st.image(file_path, caption="OCR Result", width=WIDTH)
                    button_col1, button_col2 = st.columns([1, 1])
                    with button_col1:
                        if st.button(f"View {subfolder}/{frame_idx}"):
                            play_dialog(subfolder, frame_idx, "ocr")
                    with button_col2:
                        if st.button(f"Zoom {subfolder}/{frame_idx}"):
                            zoom_image(file_path, subfolder, frame_idx, "ocr")
