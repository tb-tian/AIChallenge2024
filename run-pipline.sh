#!/bin/bash
set -euxo pipefail

# 1. extract kf
python keyframe_extractor.py

# 2. get information from video sound (whisper + translation model)
echo "video_to_audio"
python video_to_audio.py

echo "speech_to_text"
python speech_to_text_v2.py

echo "translation"
python translation.py

# 3. get visual information using CLIP
echo "keyframe_embedding"
python keyframe_embedding.py

# 4.
python mapping.py

