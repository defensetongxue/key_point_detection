import numpy as np
from .transforms_kit import *
from torchvision import transforms
from torch.utils.data import Dataset
import json
import os

class KeypointDetectionDatasetRegression(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode

        # Load annotations
        with open(os.path.join(data_path, 'annotations', f'{mode}.json'), 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        img_path = os.path.join(self.data_path, 'images', f"{annotation['image_id']}.png")
        img = Image.open(img_path).convert('RGB')
        keypoints = annotation['keypoints']

        if self.transform:
            img, labels = self.transform(img, keypoints)
        else:
            image_width, image_height = img.size
            labels = create_regression_label(keypoints, image_width,image_height)

        return img, labels

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
        labels = create_regression_label(keypoints, image_width, image_height)
        return img, labels
    

def create_regression_label(annotation, image_width, image_height):
    keypoints = np.array(annotation['keypoints'], dtype=np.float32)
    keypoints[:, 0] /= image_width
    keypoints[:, 1] /= image_height

    return keypoints