import codecs
import os
import subprocess

import librosa
import soundfile
import whisper
from tqdm import tqdm

import helpers
from helpers import get_logger
from load_all_video_keyframes_info import load_all_video_keyframes_info
from slicer import Slicer

whisper_model = whisper.load_model("large")
# whisper_model = whisper.load_model("tiny")

logger = get_logger()


def recognize_speech_from_audio(audio_path) -> str:
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


def speech_to_text(audio_path, transcript_path):
    """
    Processes a video file to extract audio and recognize speech.
    """
    logger.info(f"slicing and running whisper for {audio_path}...")

    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=500,
        hop_size=10,
        max_sil_kept=400,
    )
    chunks = slicer.slice(audio)

    transcript = ""
    for i, chunk in enumerate(tqdm(chunks, unit="chunk")):
        if len(chunk.shape) > 1:
            chunk = chunk.T

        chunk_path = os.path.join("/tmp", f"{i}.wav")
        soundfile.write(chunk_path, chunk, sr)
        text = recognize_speech_from_audio(chunk_path)
        transcript += text
        transcript += "\n"
        os.remove(chunk_path)

    with open(transcript_path, "w") as f:
        f.write(transcript)


if __name__ == "__main__":
    all_video, video_keyframe_dict = load_all_video_keyframes_info()
    for v in all_video:
        audio_path = f"./data-staging/audio/{v}.wav"
        transcript_path = f"./data-staging/transcripts/{v}.txt"

        if helpers.is_exits(transcript_path):
            logger.debug(f"ignore {transcript_path}")
            continue
        logger.info(f"running {v}...")
        speech_to_text(audio_path, transcript_path)
