import torch
import random
import numpy as np
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F
from torchvision.transforms import  Normalize, ToTensor

class KeypointDetectionTransform:
    def __init__(self, mean=[0.4623, 0.3856, 0.2822], std=[0.2527, 0.1889, 0.1334], resize=(416, 416)):
        self.mean = mean
        self.std = std
        self.resize = Resize(resize)

    def __call__(self, img, keypoints):
        img, keypoints = self.resize(img, keypoints)
        img, keypoints = Fix_RandomRotation()(img, keypoints)
        img, keypoints = RandomHorizontalFlip()(img, keypoints)
        img, keypoints = RandomVerticalFlip()(img, keypoints)
        img = ContrastEnhancement()(img)
        img = ToTensor()(img)
        img = Normalize(self.mean, self.std)(img)
        return img, keypoints
    
class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        original_size = np.array(img.size, dtype=np.float32)
        img = F.resize(img, self.size)
        new_size = np.array(img.size, dtype=np.float32)
        scale = new_size / original_size
        label[1] *= scale[0]
        label[2] *= scale[1]
        label[3] *= scale[0]
        label[4] *= scale[1]
        return img, label

class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img, label):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img, label
    
# Fix_RandomRotation
class Fix_RandomRotation:
    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def get_params(self):
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle
    def __call__(self, img, keypoints):
        angle = self.get_params()
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        keypoints = rotate_keypoints(keypoints, angle, img.size)
        return img, keypoints

# RandomHorizontalFlip
class RandomHorizontalFlip:
    def __call__(self, img, keypoints):
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
            keypoints = flip_keypoints_horizontal(keypoints, img.size[0])
        return img, keypoints

# RandomVerticalFlip
class RandomVerticalFlip:
    def __call__(self, img, keypoints):
        if torch.rand(1) < 0.5:
            img = F.vflip(img)
            keypoints = flip_keypoints_vertical(keypoints, img.size[1])
        return img, keypoints

# Helper functions for rotating, flipping keypoints horizontally and vertically
def rotate_keypoints(keypoints, angle, img_size):
    angle = np.deg2rad(angle)
    img_center = np.array(img_size) / 2
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    rotated_keypoints = (keypoints - img_center) @ rot_matrix + img_center
    return rotated_keypoints

def flip_keypoints_horizontal(keypoints, img_width):
    flipped_keypoints = keypoints.copy()
    flipped_keypoints[:, 0] = img_width - keypoints[:, 0]
    return flipped_keypoints

def flip_keypoints_vertical(keypoints, img_height):
    flipped_keypoints = keypoints.copy()
    flipped_keypoints[:, 1] = img_height - keypoints[:, 1]
    return flipped_keypoints
