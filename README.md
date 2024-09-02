## Installation

### For Ubuntu
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install. Highly recommend using python package manager like [conda](https://docs.conda.io/en/latest/)

```bash
pip install -r requirements.txt
pip install faiss-cpu
```

### For Window
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install. Highly recommend using python package manager like [conda](https://docs.conda.io/en/latest/)

```bash
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/chroma-core/chroma.git
pip install streamlit faiss-cpu open-clip-torch
```

## Adding new dependencies

recommend using https://github.com/astral-sh/uv

## Sample Dataset

Download from https://drive.google.com/drive/folders/1wzM8PtgxXgDDeQJtzGXmmEn1x43YDL9l and place inside `./datasets` directory.

```
mkdir datasets
cd datasets
# copy all downloaded data here
# then unzip all
```

your projects structure should look like this

```
.
├── cli.py
├── datasets
│   ├── clip-features
│   ├── keyframes
│   ├── map-keyframes
│   ├── texts (created for storing vietnamese and english translation)
│   ├── timestamps (created for storing sound chunk)
│   ├── videos
│   └── document_embedding_info.pkl (created after running document_embedding.py)
│   ├── info.npy (created after running keyframe_embedding.py)
│   ├── embedding.index (created after running keyframe_embedding.py)
│   └── mapping.csv (created after running mapping.py)
│   └── tfidf_matrix.npz (created after running document_embedding.py)
│   └── tfidf_vectorizer.pkl (created after running document_embedding.py)
│   └── ...
├── README.md
├── document_embedding.py
├── keyframe_embedding.py
├── loading_dict.py
├── requirements.in
├── requirements.txt
├── slicer.py
├── speech_to_text.py
├── translation.py
├── vectordb.py
├── video_processing.py
└── web_app.py

```


## Usage

### Indexing Stage

Step 1: Preparing projects structure as above

Step 2: Run video_processing.py
```bash
python video_processing.py
```

Step 3: Run document_embedding.py and keyframe_embedding.py
```bash
python document_embedding.py 
python keyframe_embedding.py
```

Step 4: Run mapping.py
```bash
python mapping.py
```

### Retrieval Stage

Openning UI
```bash
streamlit run web_app.py
```

Testing the model and sample data
```bash
python cli.py
```


# Coding style

PEP-8 with black formatter and isort

```
black .
isort .
```
