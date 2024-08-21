from vectordb import VectorDB
from PIL import Image


if __name__ == "__main__":
    vecdb = VectorDB()
    query = input("Your query term: ")
    res = vecdb.search_text(query)

    for video, keyframe, similarity in res:
        file_path = f"./datasets/keyframes/{video}/{keyframe}.jpg"
        image = Image.open(file_path)
        image.show()
        print(f"Video: {video}, Keyframe: {keyframe}, Similarity: {similarity}")
