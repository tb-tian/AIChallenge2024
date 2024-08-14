import numpy
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from glob import glob

# This python script is used to use vector database, which is chromadb instead of npy (numpy arrays)

# client = chromadb.Client()
# embedding_function = OpenCLIPEmbeddingFunction()
# data_loader = ImageLoader()
# collection = client.create_collection(
#     name = 'keyframes',
#     embedding_function=embedding_function,
#     data_loader=data_loader,
# )

# URIs of the images
uris = glob('/home/thienan/Downloads/DataSampleAIC23-20240811T084355Z-002/DataSampleAIC23/Keyframes_L01/keyframes/L01_V001/*.jpg')
ids = []
# Load images and generate embeddings
# embeddings = [embedding_function.encode_image(uri) for uri in uris]

for uri in uris:
    _, id = uri[:-4].rsplit('/',1)
    ids.append(id)

ids = sorted(ids)
print(ids)

# collection.add(
#     ids=['id1', 'id2', 'id3'],
    
# )
# print(len(embedding_function._encode_text("Hello")))
