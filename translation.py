from easynmt import EasyNMT
import codecs
from loading_dict import create_video_list_and_video_keyframe_dict
translate_model = EasyNMT("m2m_100_418M")


def translate(video):
    '''
    Translate text from vietnamese to english
    '''
    vi_text = f"./datasets/texts/{video}_vi.txt"
    en_text = f"./datasets/texts/{video}_en.txt"
    with open(vi_text, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    with open(en_text, "a") as file:
        for line in lines:
            file.write(translate_model.translate(line, target_lang="en") + "\n")


if __name__ == "__main__":
    all_video, video_keyframe_dict = create_video_list_and_video_keyframe_dict()
    for v in all_video:
        translate(v)