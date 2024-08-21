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
│   ├── videos
│   ├── info.npy (created after running embedding.py)
│   ├── embedding.index (created after running embedding.py)
│   └── ...
├── README.md
├── embedding.py
├── requirements.txt
├── web_app.py
└── vectordb.py

```


## Usage

Running embedding.py first
```bash
python embedding.py
```

Testing UI
```bash
streamlit run web_app.py
```

Testing the model and sample data
```bash
python cli.py
```


# Coding style

PEP-8 with black formatter 

```
black .
```
