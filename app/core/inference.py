import matplotlib.pyplot as plt
from pathlib import Path
import uuid
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForImageClassification
from peft import PeftModel
from core.grad_cam import ViTGradCAM

from config import CAM_IMAGEDIR, CHECKPOINT_DIR

# CAM_IMAGEDIR = "images/cam_images/"
# CHECKPOINT_DIR = "lora_adapter/best/PEFT_dino_base_l8_mixup" 
Path(CAM_IMAGEDIR).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

model_name = "facebook/dinov2-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = AutoModelForImageClassification.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR).to(device)
model.eval()

cam_model = ViTGradCAM(model, use_cuda=torch.cuda.is_available())

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_processed = transform(img).float().unsqueeze(0).to(device)
    return img_processed, img

def preprocess_predict_image(image_path):
    processed_image, original_image = preprocess_image(image_path)

    visualization, _, prediction, predicted_proba = cam_model.visualize(
        processed_image, original_image
    )

    cam_image_path = Path(CAM_IMAGEDIR) / f"{uuid.uuid4()}.jpg"
    plt.imsave(cam_image_path, visualization)

    return cam_image_path, prediction, predicted_proba





