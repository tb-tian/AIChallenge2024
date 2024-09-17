## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install. Highly recommend using python package manager like [conda](https://docs.conda.io/en/latest/)

Tested on python 3.10

(optional for conda user)

```bash
conda create -n AIChallenge2024 python=3.10
conda activate AIChallenge2024
```

install all requirements

```bash
pip install -r requirements.txt
```

optional faster mirror: https://gist.github.com/schnell18/d0ed716917905d2c142a370906cfa32f
    
## Managing dependencies

recommend using https://github.com/astral-sh/uv

```
# compile 
uv pip compile requirements.in --output-file requirements.txt
# sync
uv pip sync requirements.txt
```

## Data Folder Structure


your projects structure should look like this

```
tree -L 2 -l .

.
├── app.log
├── cli.py
├── data-index
│   └── ...
├── data-source ->
│   └── videos
├── data-staging
│   ├── audio-chunk-timestamps
│   ├── clip-features
│   ├── map-keyframes
│   ├── preprocessing
│   ├── transcripts
│   └── transcripts-en
├── document_embedding.py
├── format-code.sh
├── helpers.py
├── hybrid_search.py
├── keyframe_embedding.py
├── keyframe_extractor.py
├── loading_dict.py
├── mapping.py
├── models
├── README.md
├── requirements.in
├── requirements.txt
├── run-pipline.sh

```

1. data-source: read only folder, source data should be placed here, at the moment, it should only have source video only

```bash
mkdir data-source
# downloaded data here
wget url1 url2 url3...
unzip *.zip
mv video/* videos/
```

Download from https://drive.google.com/drive/folders/1wzM8PtgxXgDDeQJtzGXmmEn1x43YDL9l and place inside that `./data-source` directory.

2. data-staging: all transformation should go there
```bash
mkdir data-staging
mkdir data-staging/audio-chunk-timestamps
mkdir data-staging/keyframes
mkdir data-staging/preprocessing
mkdir data-staging/map-keyframes
mkdir data-staging/transcripts
mkdir data-staging/transcripts-en
mkdir data-staging/clip-features
```

3. data-index: artifact of index process go here
```bash
mkdir data-index
```

## Usage

### Indexing Stage

TLDR: using `bash run-pipline.sh`

### Retrieval Stage

Running the web_app version
```bash
streamlit run web_app.py
# or
streamlit run web_app.py --browser.serverAddress '0.0.0.0'
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

# or

./format-code.sh
```

# run on CPU

best is to use with GPU (NVIDIA) machine, but if you want to experiment on CPU machine, by default it will try to check `torch.cuda.is_available()` and use the suitable model 

