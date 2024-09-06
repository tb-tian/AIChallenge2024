import codecs
import os
import subprocess

from tqdm import tqdm

import helpers
from helpers import get_logger
from load_all_video_keyframes_info import load_all_video_keyframes_info
import whisperx

logger = get_logger()

device = "cuda"
# audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
whisperx_model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="vi")

def whisperx_speech_to_text(audio_path, video_path, transcript_path):
    audio = whisperx.load_audio(audio_path)
    result = whisperx_model.transcribe(audio, batch_size=batch_size, print_progress=True)
    # print(result["segments"])  # before alignment
    text = ""

    vid_name = os.path.basename(video_path)[:-4]
    with open(f"./data-staging/audio-chunk-timestamps/{vid_name}.csv", "w") as ts_file:
        ts_file.write("start_time,end_time\n")
        for v in result["segments"]:
            text += v["text"].strip()
            text += "\n"
            ts_file.write(f"""{v["start"]},{v["end"]}\n""")
    # print(text)

    # 2. Align whisper output
    # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # print(result["segments"])  # after alignment

    with open(transcript_path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    all_video, video_keyframe_dict = load_all_video_keyframes_info()
    for v in all_video:
        audio_path = f"./data-staging/audio/{v}.wav"
        video_path = f"./data-source/videos/{v}.mp4"
        transcript_path = f"./data-staging/transcripts/{v}.txt"

        if helpers.is_exits(transcript_path):
            logger.debug(f"ignore {transcript_path}")
            continue
        logger.info(f"running speed to text from {audio_path} to {transcript_path} ...")
        whisperx_speech_to_text(audio_path, video_path, transcript_path)
