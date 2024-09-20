import os
import pandas as pd
import easyocr
from PIL import Image, ImageFilter
import cv2
from load_all_video_keyframes_info import load_all_video_keyframes_info
import concurrent.futures


all_video, video_keyframe_dict = load_all_video_keyframes_info()
reader = easyocr.Reader(["vi", "en"], gpu=True)


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.filter(ImageFilter.SHARPEN)
    return image


def process_file(file_path):
    print(f"Processing file: {file_path}")
    # processed_image = cv2.imread(file_path)
    text = reader.readtext(file_path, detail=0, paragraph=True)
    subfolder = os.path.relpath(os.path.dirname(file_path), image_folder)
    return {
        "file_name": os.path.basename(file_path),
        "subfolder": subfolder,
        "text": " ".join(text),
    }


def perform_ocr_on_images(image_folder, output_csv):
    data = []
    print("Starting OCR process...")

    for v in all_video[:1]:
        for kf in video_keyframe_dict[v]:
            file_path = f"./data-staging/keyframes/{v}/{kf}.jpg"
            result = process_file(file_path)
            data.append(result)

    # for root, _, files in os.walk(image_folder):
    #     file_paths = [os.path.join(root, file) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
    #     for file_path in file_paths:
    #         result = process_file(file_path)
    #         data.append(result)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"OCR process completed. Results saved to {output_csv}.")


if __name__ == "__main__":
    # file_path = './data-staging/keyframes/L01_V001/0006.jpg'
    # result = reader.readtext(file_path, detail=0, paragraph=True)
    # for line in result:
    #     print(line)
    image_folder = "./data-staging/keyframes"
    output_csv = "./data-staging/ocr_results.csv"
    # perform_ocr_on_images(image_folder, output_csv)

    # with Pool(8) as p:

    with concurrent.futures.ProcessPoolExecutor() as executor:
        data_all = []
        for v in all_video:
            for kf in video_keyframe_dict[v]:
                file_path = f"./data-staging/keyframes/{v}/{kf}.jpg"
                data_all.append(file_path)
        executor.map(process_file, data_all)
