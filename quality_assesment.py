import os
import torchvision
import torch
import piq
from torchvision.transforms.functional import to_tensor, InterpolationMode
from PIL import Image
import json


folder_path = r"C:\Users\Max\Desktop\deepface_script\generatedphotos"

image_paths = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

def load_image_tensor(path):
    image = Image.open(path)
    tensor = torchvision.transforms.functional.pil_to_tensor(image)
    return tensor

@torch.no_grad()
def compute_metrics(paths):
    scores_brisque = []
    scores_clip_iqa = []

    for path in paths:
        image_tensor = load_image_tensor(os.path.join(folder_path, path)).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # Вычисление BRISQUE
        score_brisque = piq.brisque(image_tensor, data_range=255, reduction='none')
        scores_brisque.append(score_brisque.item())

        # Вычисление CLIP-IQA
        clip_iqa = piq.CLIPIQA(data_range=255).to(image_tensor.device)
        score_clip_iqa = clip_iqa(image_tensor)
        scores_clip_iqa.append(score_clip_iqa.item())
    return scores_brisque, scores_clip_iqa


brisque_scores, clip_iqa_scores = compute_metrics(image_paths)
result_dict = {key: (brisque, clip) for key, brisque, clip in zip(image_paths, brisque_scores, clip_iqa_scores)}
print(json.dumps(result_dict, indent=4))

