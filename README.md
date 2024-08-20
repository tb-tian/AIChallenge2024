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
pip install git+git+https://github.com/chroma-core/chroma.git
pip install streamlit faiss-cpu
```

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
├── compute.py
├── datasets
│   ├── AIC-2023-Baseline-v1.ipynb
│   ├── clip-features-vit-b32-sample
│   ├── clip-features-vit-b32-sample.zip
│   ├── keyframes
│   ├── Keyframes_L01.zip
│   ├── map-keyframes-sample
│   └── map-keyframes-sample.zip
├── README.md
├── requirements.txt
├── user_interface.py
└── vectordb.py

```


## Usage

Testing UI
```bash
streamlit run web_app.py
```

Testing the model and sample data
```bash
python compute.py
```


# Coding style

PEP-8 with black formater 

```
black .
```
