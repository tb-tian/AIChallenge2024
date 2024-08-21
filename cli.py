from vectordb import VectorDB
from PIL import Image


if __name__ == "__main__":
    v = VectorDB()
    query = input("Your query term: ")
    res = v.search_text(query)

    for v, k, similarity in res:
        file_path = f"./datasets/keyframes/{v}/{k}.jpg"
        image = Image.open(file_path)
        image.show()
        print(f"Video: {v}, Keyframe: {k}, Similarity: {similarity}")
