import codecs
import os
import subprocess

import librosa
import soundfile
import whisper

from loading_dict import create_video_list_and_video_keyframe_dict
from slicer import Slicer

whisper_model = whisper.load_model("large")


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
    transcript = whisper_model.transcribe(
        word_timestamps=True,
        audio=audio_path,
        fp16=False,
        compression_ratio_threshold=2.0,
    )
    text = ""
    for segment in transcript["segments"]:
        text += "".join(f"{word['word']}" for word in segment["words"])
    return text


def process_video(video_path):
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
    chunks = slicer.slice(audio,video_path)

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

        os.remove(chunk_path)

    os.remove(audio_path)
