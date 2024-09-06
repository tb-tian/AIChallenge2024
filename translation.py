import codecs

from easynmt import EasyNMT

from helpers import get_logger
from loading_dict import create_video_list_and_video_keyframe_dict

# translate_model = EasyNMT("m2m_100_418M")
translate_model = EasyNMT("opus-mt")

logger = get_logger()


def translate(video):
    """
    Translate text from vietnamese to english
    """
    logger.info(f"translating {video}...")
    vi_text = f"./data-staging/transcripts/{video}.txt"
    en_text = f"./data-staging/transcripts-en/{video}.txt"
    with open(vi_text, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(en_text, "a") as file:
        for line in lines:
            file.write(translate_model.translate(line, target_lang="en"))


if __name__ == "__main__":
    all_video, video_keyframe_dict = create_video_list_and_video_keyframe_dict()
    for v in all_video:
        translate(v)
