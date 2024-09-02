from glob import glob


def create_video_list_and_video_keyframe_dict():
    """
    Create a list of video and list of keyframe
    """
    all_keyframe = glob("./datasets/keyframes/*/*.jpg")
    video_keyframe_dict = {}
    all_video = glob("./datasets/videos/*")
    all_video = [v.rsplit("/", 1)[-1][:-4] for v in all_video]
    all_video = sorted(all_video)

    print(f"loaded {len(all_keyframe)} keyframes")
    print(f"loaded {len(all_video)} videos")

    for kf in all_keyframe:
        _, vid, kf = kf[:-4].rsplit("/", 2)
        if vid not in video_keyframe_dict.keys():
            video_keyframe_dict[vid] = [kf]
        else:
            video_keyframe_dict[vid].append(kf)

    for k, v in video_keyframe_dict.items():
        video_keyframe_dict[k] = sorted(v)

    return all_video, video_keyframe_dict