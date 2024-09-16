import os
import subprocess

import helpers
from helpers import get_logger
from load_all_video_keyframes_info import load_all_video_keyframes_info

logger = get_logger()


def video_to_audio(in_path, out_path):
    logger.debug(f"converting video {in_path} to {out_path}")
    command = f"ffmpeg -i {in_path} -vf scale=240:-2 -c:v libx264 -crf 25 -preset medium -c:a copy {out_path}"
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    all_video, video_keyframe_dict = load_all_video_keyframes_info()
    for v in all_video:
        input_video = f"./data-source/videos/{v}.mp4"
        output_video = os.path.basename(input_video)
        output_video = f"./data-staging/videos-preview/{output_video[:-4]}.mp4"

        if helpers.is_exits(output_video):
            logger.debug(f"ignore {output_video}")
            continue
        video_to_audio(input_video, output_video)
