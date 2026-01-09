
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

TEST_IMAGEDIR = BASE_DIR / "images" / "test_images" #images uploaded to the server
CAM_IMAGEDIR = BASE_DIR / "images" / "cam_images" #Class Activation Map images generated at inference time

CHECKPOINT_DIR = BASE_DIR / "lora_adapter/best/PEFT_dino_base_l8_mixup" #Directory for our fine tuned adapters