import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms
from .transforms_kit import *

class KeypointDetectionDatasetHeatmap(Dataset):
    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode

        # Load annotations
        self.annotations = json.load(open(os.path.join(data_path, 'annotations', f"{mode}.json")))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        img_path = os.path.join(self.data_path, 'images', f"{annotation['image_id']}.png")
        img = Image.open(img_path).convert('RGB')
        keypoints = annotation['keypoints']

        if self.transform:
            img, heatmap = self.transform(img, keypoints)
        else:
            image_width, image_height = img.size
            heatmap = create_heatmap_label(keypoints, image_width,image_height)

        return img, heatmap


class KeypointDetectionTransformHeatmap:
    def __init__(self, mean=[0.4623, 0.3856, 0.2822],
                 std=[0.2527, 0.1889, 0.1334],
                 size=(416, 416),
                 mode='train'):
        self.size = size
        self.mode = mode

        if self.mode == 'train':
            self.transforms = transforms.Compose([
                ContrastEnhancement(factor=1.5),
                Resize(size),
                Fix_RandomRotation(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transforms = transforms.Compose([
                ContrastEnhancement(factor=1.5),
                Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __call__(self, img, keypoints):
        img, keypoints = self.transforms(img, keypoints)
        image_width, image_height = img.size
        heatmap = create_heatmap_label(keypoints, image_width, image_height)
        return img, heatmap
    

def create_heatmap_label(annotation,
                            output_width, output_height,
                              sigma=2):
    num_keypoints = len(annotation['keypoints'])
    heatmap = np.zeros((num_keypoints, 
                        output_height, output_width), dtype=np.float32)

    for i, (x, y) in enumerate(annotation['keypoints']):
        
        # Create a Gaussian heatmap around the keypoint location
        xx, yy = np.meshgrid(np.arange(output_width), np.arange(output_height),
                              sparse=True)
        heatmap[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return heatmap
