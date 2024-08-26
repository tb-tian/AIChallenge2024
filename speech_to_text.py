import speech_recognition as srr
import subprocess
import os
import numpy as np
import librosa
import soundfile
from slicer import Slicer
import codecs
from easynmt import EasyNMT
import whisper

recognizer = whisper.load_model("large")
model = EasyNMT("m2m_100_418M")


def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file using ffmpeg.
    """
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    subprocess.call(command, shell=True)


def recognize_speech_from_audio(audio_path):
    """
    Recognizes speech from an audio file using the whisper library.
    """
    transcript = recognizer.transcribe(
        word_timestamps=True,
        audio=audio_path,
        fp16=False,
        compression_ratio_threshold=2.0,
    )
    text = ""
    for segment in transcript["segments"]:
        text += "".join(f"{word['word']}" for word in segment["words"])
    return text


def process_video(video_path, initialdir):
    """
    Processes a video file to extract audio and recognize speech.
    """
    audio_path = f"{video_path[:-4]}.wav"
    extract_audio_from_video(video_path, audio_path)
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

    video_name = os.path.basename(video_path)[:-4]
    output_dir = "./datasets/texts"
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T

        chunk_path = os.path.join(output_dir, f"{i}.wav")
        soundfile.write(chunk_path, chunk, sr)

        text = recognize_speech_from_audio(chunk_path)

        with codecs.open(f"{output_dir}/{video_name}_vi.txt", "a", "utf-8") as f:
            f.write(text + "\n")

        with codecs.open(f"{output_dir}/{video_name}_en.txt", "a", "utf-8") as f:
            f.write(model.translate(text, target_lang="en") + "\n")

        os.remove(chunk_path)

    os.remove(audio_path)


# Usage
if __name__ == "__main__":
    initialdir = os.getcwd()
    video_path = "./datasets/videos/L01_V001.mp4"
    # video_path = "test.mp4"
    process_video(video_path, initialdir)
