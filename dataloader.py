from typing import Tuple
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils import load_image, load_mask


class FlameDataset(Dataset):
    def __init__(self, base_dir: str, image_transform: transforms.Compose, mask_transform: transforms.Compose):
        self._images_dir = os.path.join(base_dir, "Images")
        self._masks_dir = os.path.join(base_dir, 'Masks')
        self._num_images = len(os.listdir(self._images_dir))
        self._image_transform = image_transform
        self._mask_transform = mask_transform

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self._images_dir, f"image_{idx}.jpg")
        mask_path = os.path.join(self._masks_dir, f"image_{idx}.png")
        image = load_image(image_path)
        image = self._image_transform(image)
        mask = load_mask(mask_path)
        mask = self._mask_transform(mask)
        mask = torch.squeeze(mask)
        return image, mask
