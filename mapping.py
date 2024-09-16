import csv
import math
import os

from helpers import get_logger
from load_all_video_keyframes_info import load_all_video_keyframes_info

all_video, video_keyframe_dict = load_all_video_keyframes_info()
logger = get_logger()


# TODO: check round up! timestamp
def mapping_from_keyframe_to_audio_chunk(video_id):
    # determine which audio chunk would the keyframe belongs to
    keyframe_path = f"./data-staging/map-keyframes/{video_id}.csv"
    chunk_path = f"./data-staging/audio-chunk-timestamps/{video_id}.csv"
    tmp_vid_mapping = "/tmp/mapping.csv"
    video_keyframe_chunk_dict = {video_id: {}}

    with open(keyframe_path) as keyframe_file, open(chunk_path) as chunk_file, open(
        tmp_vid_mapping, "w"
    ) as mapping_file:
        keyframe_time = csv.reader(keyframe_file)
        chunk_time = csv.reader(chunk_file)
        mapping = csv.writer(mapping_file)

        next(keyframe_time)
        next(chunk_time)

        keyframe_time = list(keyframe_time)
        chunk_time = list(chunk_time)
        mapping.writerow(["keyframe_id", "chunk"])

        for kf_row in keyframe_time:
            n, pts_time, _, _ = kf_row
            pts_time = float(pts_time)

            for i, ch_row in enumerate(chunk_time):
                start_time, end_time = map(float, ch_row)
                if start_time <= pts_time and pts_time <= end_time:
                    mapping.writerow([n, i])
                    video_keyframe_chunk_dict[video_id][
                        video_keyframe_dict[video_id][int(n) - 1]
                    ] = i
                    break
                elif pts_time < start_time:
                    mapping.writerow([n, i - 1])
                    video_keyframe_chunk_dict[video_id][
                        video_keyframe_dict[video_id][int(n) - 1]
                    ] = (i - 1)
                    break

    os.remove(tmp_vid_mapping)
    return video_keyframe_chunk_dict


def main():
    output_mapping_path = "./data-index/mapping.csv"

    with open(output_mapping_path, "w") as file:
        file = csv.writer(file)
        file.writerow(["video", "keyframe", "chunk"])
        for v in all_video:
            logger.info(f"processing {v}...")
            video_keyframe_chunk_dict = mapping_from_keyframe_to_audio_chunk(v)
            for kf in video_keyframe_dict[v]:
                chunk = video_keyframe_chunk_dict[v].get(kf)
                if chunk is not None:
                    file.writerow([v, kf, chunk])
                    prev_chunk = chunk
                else:
                    file.writerow([v, kf, prev_chunk])


if __name__ == "__main__":
    main()
