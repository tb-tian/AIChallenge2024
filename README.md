## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install. Highly recommend using python package manager like [conda](https://docs.conda.io/en/latest/)

```bash
pip install -r requirements.txt
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

```


## Usage

Testing UI
```bash
streamlit run user_interface.py
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
