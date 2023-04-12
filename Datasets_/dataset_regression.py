import numpy as np
from .transforms_kit import *
from torchvision import transforms
from torch.utils.data import Dataset
import json
import os

class KeypointDetectionDatasetRegression(Dataset):
    def __init__(self, data_path, spilt='train'):
        self.data_path = data_path
        if spilt=='train':
            self.transform=KeypointDetectionTransformRegression(mode='train')
        elif spilt=='valid' or spilt=='test':
            self.transform=KeypointDetectionTransformRegression(mode='val')
        else:
            raise ValueError(
                f"Invalid spilt: {spilt}, spilt should be one of train|valid|test")
        # Load annotations
        with open(os.path.join(data_path, 'annotations', f'{spilt}.json'), 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        img_path = os.path.join(self.data_path, 'images', f"{annotation['image_id']}.png")
        img = Image.open(img_path).convert('RGB')

        if annotation['num_keypoints'] == 0:
            keypoints = torch.zeros(1, 2)  # Placeholder values, since there's no keypoint.
            presence = torch.tensor([[0]])  # Ground truth for keypoint presence (0 for no keypoint).
        
        else:
            keypoints = torch.tensor(annotation['keypoints'], dtype=torch.float32).view(-1, 3)
            keypoints = keypoints[:, :2].flatten()
            presence = torch.ones((1,1)).flatten()

        img, labels = self.transform(img, keypoints)
        # image_width, image_height = img.size
        # labels = create_regression_label(keypoints, image_width,image_height)

        return img, labels, presence

class KeypointDetectionTransformRegression:
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