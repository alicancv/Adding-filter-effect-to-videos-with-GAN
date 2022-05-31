import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

epoch = 100
learning_rate = 2e-4
batch_size = 16
l1_lambda = 100
num_workers = 2


image_size = 256
image_channels = 3
load_model = True
save_model = False
discriminator_model_name = "filter_discriminator_model.pt"
generator_model_name = "filter_generator_model_100_epoch.pt"
train_dir = "filter_dataset/train_deneme"
validation_dir = "filter_dataset/validation_deneme"

input_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2()
    ]
)

output_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2()
    ]
)

video_input_transform = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2()
    ]
)