import os
import subprocess

import helpers
from helpers import get_logger
from loading_dict import create_video_list_and_video_keyframe_dict

logger = get_logger()


def video_to_audio(video_path, audio_path):
    """
    Extracts audio from a video file using ffmpeg, silent, GPU-accelerated mode
    """
    logger.debug(f"converting video {video_path} to {audio_path}")
    command = f"ffmpeg -hide_banner -loglevel error -hwaccel cuda -y -i {video_path} -acodec pcm_s16le -ac 1 -ar 16000 -vn {audio_path}"
    # command = f"ffmpeg -y -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    all_video, video_keyframe_dict = create_video_list_and_video_keyframe_dict()
    for v in all_video:
        input_video_path = f"./data-source/videos/{v}.mp4"
        target_audio_path = os.path.basename(input_video_path)
        target_audio_path = f"./data-staging/audio/{target_audio_path[:-4]}.wav"

        if helpers.is_exits(target_audio_path):
            logger.debug(f"ignore {target_audio_path}")
            continue
        video_to_audio(input_video_path, target_audio_path)
