import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from load_all_video_keyframes_info import load_all_video_keyframes_info
from helpers import get_logger

logger = get_logger()


# # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# # this also loads the associated image processors
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="base_coco", is_eval=True, device=device
# )


class QAEngine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.txt_processors = (
            load_model_and_preprocess(
                name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device
            )
        )

    def ask(self, image_path: str, question: str) -> str:
        raw_image = Image.open(image_path).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        question = self.txt_processors["eval"](question)
        ans = self.model.predict_answers(
            samples={"image": image, "text_input": question},
            inference_method="generate",
        )
        logger.debug(f"qna test: {image_path} {question} {ans}")
        return ans


qa_engine = QAEngine()


if __name__ == "__main__":
    qna = QAEngine()
    _, video_keyframe_dict = load_all_video_keyframes_info()
    for vid, kfs in video_keyframe_dict.items():
        for kf in kfs:
            keyframe_path = f"./data-source/keyframes/{vid}/{kf}.jpg"

            raw_image = Image.open(keyframe_path).convert("RGB")

            question = "How many people in this picture"
            image = qna.vis_processors["eval"](raw_image).unsqueeze(0).to(qna.device)
            question = qna.txt_processors["eval"](question)
            ans = qna.model.predict_answers(
                samples={"image": image, "text_input": question},
                inference_method="generate",
            )
            logger.info(f"qna test: {keyframe_path} {question} {ans}")
