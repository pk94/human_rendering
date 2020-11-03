from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.texture_images = os.listdir(f'{image_dir}/textures')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(os.listdir(f'{self.image_dir}/textures'))

    def __getitem__(self, index):
        texture_image = self.texture_images[index]
        image_path = os.path.join(f'{self.image_dir}/textures', texture_image)

        image = Image.open(image_path).convert('RGB')

        image = np.array(image)

        image = self.transform(image)

        return image
