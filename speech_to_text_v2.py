import codecs
import os
import subprocess

import librosa
import soundfile
import whisper
from faster_whisper import WhisperModel
from tqdm import tqdm

import helpers
from helpers import get_logger
from loading_dict import create_video_list_and_video_keyframe_dict
from slicer import Slicer

# whisper_model = whisper.load_model("large")
whisper_model = whisper.load_model("tiny")

logger = get_logger()

model_size = "large-v3"
# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")


def recognize_speech_from_audio(audio_path) -> str:
    """
    Recognizes speech from an audio file using the whisper library.
    """
    logger.debug(">>>")
    segments, info = model.transcribe(
        audio=audio_path, beam_size=5, compression_ratio_threshold=2.0, language="vi"
    )
    ret = ""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        ret += segment.text
    return ret
    # # transcript = whisper_model.transcribe(
    # #     word_timestamps=True,
    # #     audio=audio_path,
    # #     fp16=False,
    # #     compression_ratio_threshold=2.0,
    # #     language="vietnamese",2
    # # )
    # text = ""
    # # for segment in transcript["segments"]:
    # #     text += "".join(f"{word['word']}" for word in segment["words"])
    # # # print(text)
    # # return text
    #
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    #
    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    #     text += "".join(f"{word['word']}" for word in segment["words"])
    # return text


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

    # video_name = os.path.basename(video_path)[:-4]
    # transcripts_output_dir = "/tmp"
    # os.makedirs(transcripts_output_dir, exist_ok=True)

    transcript = ""
    for i, chunk in enumerate(tqdm(chunks, unit="chunk")):
        if len(chunk.shape) > 1:
            chunk = chunk.T

        chunk_path = os.path.join("./data-staging", f"{i}.wav")
        soundfile.write(chunk_path, chunk, sr)
        text = recognize_speech_from_audio(chunk_path)
        transcript += text
        transcript += "\n"

        # with codecs.open(
        #     f"{transcripts_output_dir}/{video_name}.txt", "a", "utf-8"
        # ) as f:
        #     f.write(text + "\n")
        os.remove(chunk_path)

    with open(transcript_path, "w") as f:
        f.write(transcript)


if __name__ == "__main__":
    all_video, video_keyframe_dict = create_video_list_and_video_keyframe_dict()
    for v in all_video:
        logger.info(f"running {v}...")
        audio_path = f"./data-staging/audio/{v}.wav"
        transcript_path = f"./data-staging/transcripts/{v}.txt"

        if helpers.is_exits(transcript_path):
            logger.debug(f"ignore {transcript_path}")
            continue
        speech_to_text(audio_path, transcript_path)
