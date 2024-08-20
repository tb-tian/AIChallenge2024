import streamlit as st
import numpy as np
import clip
import torch
import os
from PIL import Image
from io import BytesIO
import base64

from vectordb import VectorDB

WIDTH = 350

if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = VectorDB()
vectordb = st.session_state["vectordb"]

if __name__ == "__main__":
    # st.set_page_config(layout="wide")
    st.set_page_config(page_title="TIAN™ Video Search", page_icon="🔍", layout="wide")

    st.header("TIAN™ Video Search")
    st.write(
        "Welcome to TIAN Video Search. You can blah blah blah here. And blah blah blah there also."
    )
    search_term = st.text_input("Search: ")

    print("searching...", search_term)

    res = vectordb.search_text(search_term)

    # Create columns for layout
    col1, col2, col3, col4 = st.columns(4)
    

    # Assign results to columns
    for i, (v, k, similarity) in enumerate(res):
        file_path = f"./datasets/keyframes/{v}/{k}.jpg"
        
        # Use column assignment based on index
        if i % 4 == 0:
            col = col1
        elif i % 4 == 1:
            col = col2
        elif i % 4 == 2:
            col = col3
        else:
            col = col4

        with open(file_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
        
        # Display the HTML content
        with col:
            st.html(
                f"""
                <div style="width: {WIDTH}px; text-align: center;">
                    <img src="data:image/gif;base64, {data}" style="width: {WIDTH}px; height: {WIDTH}px; object-fit: cover;">
                    <p>do some thing :v</p>
                </div>
                """
            )