import csv
import os
import streamlit as st

ocr_results_file = './data-staging/ocr_results_en.csv'
map_keyframes_folder = './data-staging/map-keyframes'

def load_csv_file(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: {file_path} does not exist.")
        return []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]

def find_match_in_ocr(input_text, ocr_data):
    results = []
    for row in ocr_data:
        if input_text.lower() in row['text'].lower():
            results.append(row)
    return results

def find_frame_idx(subfolder, file_name):
    map_keyframes_file = os.path.join(map_keyframes_folder, f"{subfolder}.csv")
    keyframes_data = load_csv_file(map_keyframes_file)

    if not keyframes_data:
        return None

    file_number = int(file_name.split('.')[0])
    file_number_adjusted = file_number + 1

    for row in keyframes_data:
        if int(row['n']) == file_number_adjusted:
            return row['frame_idx']
    return None

def display_search_results(input_text, kf, video):
    ocr_data = load_csv_file(ocr_results_file)
    if not kf:
        st.info("No frame index found.")
        return
    
    print(f"Searching for {kf}")
    kf_with_extension = f"{kf}.jpg"
    for row in ocr_data:
        if kf_with_extension in row['file_name'] and video in row['subfolder']:
            highlighted_text = row["text"].replace(input_text, f"<span style='color: yellow;'>{input_text}</span>")
            st.write(highlighted_text, unsafe_allow_html=True)
    
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
        subfolder = match['subfolder']
        file_name = match['file_name']
        frame_idx = find_frame_idx(subfolder, file_name)
        if frame_idx:
            result_data.append((subfolder, file_name, frame_idx, 0.0))  
    return result_data