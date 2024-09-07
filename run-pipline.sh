#!/bin/bash
set -euxo pipefail

# 1. get information from video sound (whisper + translation model)
echo "video_to_audio"
python video_to_audio.py

echo "speech_to_text"
python speech_to_text_v2.py

echo "translation"
python translation.py

# 2. get visual information using CLIP
echo "keyframe_embedding"
python keyframe_embedding.py

# 3.
python mapping.py

