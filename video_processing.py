from loading_dict import create_video_list_and_video_keyframe_dict
from speech_to_text import process_video
from translation import translate

if __name__ == "__main__":
    """
    Including speech to text model and translate model
    """
    all_video, video_keyframe_dict = create_video_list_and_video_keyframe_dict()
    for v in all_video:
        video_path = f"./datasets/videos/{v}.mp4"
        process_video(video_path)
        translate(v)
