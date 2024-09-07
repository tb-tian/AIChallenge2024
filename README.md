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
│   ├── embedding.index
│   └── embedding_info.npy
├── data-source ->
│   ├── keyframes
│   └── videos
├── data-staging
│   ├── audio
│   ├── clip-features
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

1. data-source: read only folder, source data should be placed here
```bash
mkdir data-source
# copy all downloaded data here
# then unzip all
# rearrange input data to correct structure
```

Download from https://drive.google.com/drive/folders/1wzM8PtgxXgDDeQJtzGXmmEn1x43YDL9l and place inside that `./data-source` directory.

2. data-staging: all transformation should go there
```bash
mkdir -p data-staging/audio
mkdir -p data-staging/audio-chunk-timestamps
mkdir -p data-staging/clip-features
mkdir -p data-staging/transcripts
mkdir -p data-staging/transcripts-en
```

3. data-index: artifact of index process go here
```bash
mkdir data-index
```

## Usage

### Indexing Stage

TLDR: using `bash run-pipline.sh`

```
# maybe out of date docs

**Step 1**: Prepare the project structure as shown above and ensure you have the videos downloaded.

**Step 2**: Run [video_processing.py](./video_processing.py). This process may take a significant amount of time.
python video_processing.py
Then, you will have vietnamese and english dialogue in [timestamps](./datasets/timestamps/) folder.

**Step 3**: Run [keyframe_extractor.py](./keyframe_extractor.py). This process will
1. extract keyframe from the video and store them in `keyframes`
2. create a file csv of frame index in `map-keyframes` folder.
python keyframe_extractor.py



**Step 4**: Run [document_embedding.py](./document_embedding.py) and [keyframe_embedding.py](./keyframe_embedding.py)
python document_embedding.py 
python keyframe_embedding.py

**Step 5**: Run [mapping.py](./mapping.py). This process will create `mapping.csv` to map each keyframe to the corresponding chunk of dialogue.
python mapping.py
```


### Retrieval Stage

Running the web_app version
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

# or

./format-code.sh
```

# run on CPU

best is to use with GPU (NVIDIA) machine, but if you want to experiment on CPU machine, by default it will try to check `torch.cuda.is_available()` and use the suitable model 

