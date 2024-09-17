import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from helpers import get_logger
from legacy.transnetv2 import TransNetV2
from load_all_video_keyframes_info import load_all_video_keyframes_info

logger = get_logger()
all_video, video_keyframe_dict = load_all_video_keyframes_info()

trans_net_model = TransNetV2("./transnetv2-weights")


def segmentation(v):
    source_video = f"./data-source/videos/{v}.mp4"
    destination = f"./data-staging/preprocessing/"
    if os.path.exists(destination + f"{v}_predictions.txt") or os.path.exists(
        destination + f"{v}_scenes.txt"
    ):
        print(
            f"[TransNetV2] {v}_predictions.txt or {v}_scenes.txt already exists. "
            f"Skipping video {v}.",
        )
        return

    (
        video_frames,
        single_frame_predictions,
        all_frame_predictions,
    ) = trans_net_model.predict_video(source_video)

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    np.savetxt(destination + f"{v}_predictions.txt", predictions, fmt="%.6f")

    scenes = trans_net_model.predictions_to_scenes(single_frame_predictions)
    np.savetxt(destination + f"{v}_scenes.txt", scenes, fmt="%d")

    if os.path.exists(destination + f"{v}_vis.png"):
        print(
            f"[TransNetV2] {v}.vis.png already exists. "
            f"Skipping visualization of video {v}.",
            file=sys.stderr,
        )

    pil_image = trans_net_model.visualize_predictions(
        video_frames, predictions=(single_frame_predictions, all_frame_predictions)
    )
    pil_image.save(destination + f"{v}_vis.png")


def keyframe_extractor(v):
    video_path = f"./data-source/videos/{v}.mp4"
    file_path = f"./data-staging/preprocessing/{v}_scenes.txt"
    kf_path = f"./data-staging/keyframes/{v}"
    map_path = f"./data-staging/map-keyframes/{v}.csv"

    cap = cv2.VideoCapture(video_path)

    if Path(kf_path).is_dir():
        logger.info(f"{kf_path} exist, ignore...")
        return

    os.makedirs(kf_path)

    with open(map_path, "w") as mapping_file, open(file_path, "r") as file:
        mapping = csv.writer(mapping_file)
        mapping.writerow(["n", "pts_time", "fps", "frame_idx"])
        lines = file.readlines()
        for i, line in enumerate(lines):
            left, right = line.split(" ")
            mid = (int(left) + int(right)) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if ret:
                keyframe_path = f"./data-staging/keyframes/{v}/{i+1:04}.jpg"
                cv2.imwrite(keyframe_path, frame)
                fps = cap.get(cv2.CAP_PROP_FPS)
                mapping.writerow([i + 1, mid / fps, fps, mid])
                print(f"Saved keyframe {mid} of video {v}")
            else:
                print(
                    f"Failed to extract frame {mid} from video {v}.",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    os.makedirs("./data-staging/map-keyframes", exist_ok=True)

    logger.info("segmentation")
    for v in all_video:
        segmentation(v)

    logger.info("keyframe_extractor")
    for v in all_video:
        keyframe_extractor(v)
