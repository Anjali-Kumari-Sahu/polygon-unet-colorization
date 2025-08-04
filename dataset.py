import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PolygonColorDataset(Dataset):
    def __init__(self, json_path, input_dir, output_dir, color_to_idx, image_size=128, augment=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.color_to_idx = color_to_idx
        self.image_size = image_size
        self.augment = augment

        base_transforms = [T.Resize((image_size, image_size))]
        if augment:
            base_transforms += [
                T.RandomRotation(30),
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0))
            ]
        self.input_transform = T.Compose(base_transforms + [T.Grayscale(num_output_channels=1), T.ToTensor()])
        self.output_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        in_name = entry['input']
        color_name = entry['color']
        out_name = entry['output']
        in_path = os.path.join(self.input_dir, in_name)
        out_path = os.path.join(self.output_dir, out_name)
        in_img = Image.open(in_path).convert('RGB')
        out_img = Image.open(out_path).convert('RGB')
        in_tensor = self.input_transform(in_img)
        out_tensor = self.output_transform(out_img)
        color_idx = torch.tensor(self.color_to_idx[color_name], dtype=torch.long)
        return in_tensor, color_idx, out_tensor
