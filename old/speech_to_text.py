import codecs
import os
import subprocess

import librosa
import soundfile
import whisper
from loguru import logger
from tqdm import tqdm

from load_all_video_keyframes_info import load_all_video_keyframes_info
from slicer import Slicer

whisper_model = whisper.load_model("large")


def recognize_speech_from_audio(audio_path):
    """
    Recognizes speech from an audio file using the whisper library.
    """
    transcript = whisper_model.transcribe(
        word_timestamps=True,
        audio=audio_path,
        fp16=False,
        compression_ratio_threshold=2.0,
        language="vietnamese",
    )
    text = ""
    for segment in transcript["segments"]:
        text += "".join(f"{word['word']}" for word in segment["words"])
    # print(text)
    return text


def speech_to_text(video_path):
    """
    Processes a video file to extract audio and recognize speech.
    """
    logger.info(f"processing video {video_path}...")

    # target_audio_path = os.path.basename(video_path)
    # target_audio_path = f"./data-staging/audio/{target_audio_path[:-4]}.wav"
    #
    # video_to_audio(video_path, target_audio_path)
    logger.info(f"slicing and running whisper...")
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=500,
        hop_size=10,
        max_sil_kept=400,
    )
    chunks = slicer.slice(audio, video_path)

    video_name = os.path.basename(video_path)[:-4]
    transcripts_output_dir = "./data-staging/transcripts"
    os.makedirs(transcripts_output_dir, exist_ok=True)

    for i, chunk in enumerate(tqdm(chunks, unit="chunk")):
        if len(chunk.shape) > 1:
            chunk = chunk.T

        chunk_path = os.path.join("./data-staging", f"{i}.wav")
        soundfile.write(chunk_path, chunk, sr)

        text = recognize_speech_from_audio(chunk_path)

        with codecs.open(
            f"{transcripts_output_dir}/{video_name}.txt", "a", "utf-8"
        ) as f:
            f.write(text + "\n")

        os.remove(chunk_path)


if __name__ == "__main__":
    """
    Including speech to text model and translate model
    """
    all_video, video_keyframe_dict = load_all_video_keyframes_info()
    start_process = False
    for v in all_video:
        video_path = f"./datasets/videos/{v}.mp4"
        print(f"video {v} with path {video_path}")
        speech_to_text(video_path)
        # translate(v)
