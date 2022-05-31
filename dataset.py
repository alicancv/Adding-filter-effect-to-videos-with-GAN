import os
import numpy as np
import config as cfg
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.root_dir, image_name)
        image = np.array(Image.open(image_path))
        input_image = image[:, :256, :]
        target_image = image[:, 256:, :]

        input_image = cfg.input_transform(image=input_image)["image"]
        target_image = cfg.output_transform(image=target_image)["image"]

        return input_image, target_image