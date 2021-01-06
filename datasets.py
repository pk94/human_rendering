import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from random import shuffle


class VideoDataset(Dataset):
    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.texture_images = os.listdir(f'{image_dir}/textures')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
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


class DeepFashionDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.files_list = []
        for path, subdirs, files in os.walk(self.data_dir):
            for filenamename in files:
                if filenamename.endswith('h5'):
                    self.files_list.append(Path(os.path.join(path, filenamename)))
        shuffle(self.files_list)
        # self.files_list = self.files_list[:1000]


    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        sample_path = self.files_list[index]
        sample_id_path = sample_path.parent
        person_id = int(str(sample_path.name).split('_')[0])
        target_files = [x for x in sample_id_path.iterdir() if x.is_file()]
        target_person_id = 0
        target_file_path = ''
        while target_person_id != person_id:
            target_file_path = random.choice(target_files)
            target_person_id = int(str(target_file_path.name).split('_')[0])
        # target_file_path = sample_path
        sample_image, sample_instances, sample_textures, sample_uv = self.load_h5_file(sample_path)
        target_image, target_instances, target_textures, target_uv = self.load_h5_file(target_file_path)
        perm = torch.LongTensor(np.array([2, 1, 0]))
        sample_textures = sample_textures[perm, :, :]
        target_textures = target_textures[perm, :, :]
        sample_dict = {
            'image': sample_image,
            'instances': sample_instances,
            'texture': sample_textures,
            'uv': sample_uv
        }
        target_dict = {
            'image': target_image,
            'instances': target_instances,
            'texture': target_textures,
            'uv': target_uv
        }
        return {
            'sample': sample_dict,
            'target': target_dict
        }

    def load_h5_file(self, path):
        with h5py.File(path, mode="r") as h5_file:
            frame = torch.from_numpy(h5_file["frame"][:].astype(np.float32)).permute((2, 0, 1)).true_divide(255).mul(2).sub(1)
            instances = torch.from_numpy(h5_file["i"][:].astype(np.float32))
            texture = torch.from_numpy(h5_file["texture"][:].astype(np.float32)).permute((2, 0, 1)).true_divide(255).mul(2).sub(1)
            uv = torch.from_numpy(h5_file["uv"][:].astype(np.float32)).permute((2, 0, 1))
        return frame, instances, texture, uv
