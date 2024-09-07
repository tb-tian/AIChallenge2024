from easynmt import EasyNMT
from tqdm import tqdm
import nltk

import helpers
from helpers import get_logger, is_on_cpu
from load_all_video_keyframes_info import load_all_video_keyframes_info

nltk.download("punkt_tab")

if is_on_cpu():
    translate_model = EasyNMT("opus-mt")
else:
    translate_model = EasyNMT("m2m_100_418M")


logger = get_logger()


def translate(vi_text, en_text):
    """
    Translate text from vietnamese to english
    """
    with open(vi_text, "r", encoding="utf-8") as file:
        lines = file.readlines()
    tmp = ""
    for line in tqdm(lines):
        tmp += translate_model.translate(line, target_lang="en").strip()
        tmp += "\n"
    with open(en_text, "a") as file:
        file.write(tmp)


if __name__ == "__main__":
    all_video, video_keyframe_dict = load_all_video_keyframes_info()
    for video in all_video:
        vi_text = f"./data-staging/transcripts/{video}.txt"
        en_text = f"./data-staging/transcripts-en/{video}.txt"
        if helpers.is_exits(en_text):
            logger.debug(f"ignore {vi_text}")
            continue
        logger.info(f"translating {vi_text} to {en_text}...")
        translate(vi_text, en_text)
