import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from load_all_video_keyframes_info import load_all_video_keyframes_info
from helpers import get_logger

logger = get_logger()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# # this also loads the associated image processors
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="base_coco", is_eval=True, device=device
# )


model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_vqa", model_type="vqav2", is_eval=True, device=device
)

if __name__ == "__main__":
    _, video_keyframe_dict = load_all_video_keyframes_info()
    for vid, kfs in video_keyframe_dict.items():
        for kf in kfs:
            keyframe_path = f"./data-source/keyframes/{vid}/{kf}.jpg"

            raw_image = Image.open(keyframe_path).convert("RGB")
            question = "How many people in this picture"
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            question = txt_processors["eval"](question)
            ans = model.predict_answers(
                samples={"image": image, "text_input": question},
                inference_method="generate",
            )
            logger.info(f"qna test: {keyframe_path} {question} {ans}")
