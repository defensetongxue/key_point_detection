import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms
from .transforms_kit import *
class KeypointDetectionDatasetHeatmap(Dataset):
    def __init__(self, data_path, split='train',heatmap_rate=0.25,sigma=1.5):
        self.data_path = data_path
        if split=='train':
            self.transform=KeypointDetectionTransformHeatmap(mode='train')
        elif split=='valid' or split=='test':
            self.transform=KeypointDetectionTransformHeatmap(mode='val')
        else:
            raise ValueError(
                f"Invalid split: {split}, split should be one of train|valid|test")

        # Load annotations
        self.annotations = json.load(open(os.path.join(data_path, 
                                                       'annotations', f"{split}.json")))
        self.heatmap_ratio=heatmap_rate
        self.sigma=sigma

    def __len__(self):
        return len(self.annotations)
    
    def generate_target(self, img, pt, sigma, label_type='Gaussian', img_path=''):
        # Check that any part of the Gaussian is in-bounds
        tmp_size = sigma * 3
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    
        # Adjust the bounds to fit within the image dimensions
        ul[0] = max(0, ul[0])
        ul[1] = max(0, ul[1])
        br[0] = min(img.shape[1], br[0])
        br[1] = min(img.shape[0], br[1])
    
        # Generate Gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
    
        if label_type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
    
        # Usable Gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])
    
        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    
    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data=self.data_list[data_name]

        img = Image.open(data['image_path']).convert('RGB')
        
        
        keypoints = torch.tensor(data['optic_disc_gt']['position'], dtype=torch.float32).view(-1, 2)
        keypoints = keypoints[:, :2].flatten()
        # keypoint in each image
        img, keypoints = self.transform(img, keypoints)
        img_width,img_height=img.shape[1],img.shape[2]
        heatmap_width=int(img_width*self.heatmap_ratio)
        heatmap_height=int(img_height*self.heatmap_ratio)
        heatmap=np.zeros((heatmap_width,
                                   heatmap_height),dtype=np.float32)
        heatmap=self.generate_target(heatmap,keypoints*self.heatmap_ratio,sigma=self.sigma,img_path=data["image_path"])
        # labels = create_heatmap_label(keypoints, image_width,image_height)
        heatmap=heatmap[np.newaxis,:]
        return img, heatmap,data["image_path"]
        


class KeypointDetectionTransformHeatmap:
    def __init__(self, mean=[0.4623, 0.3856, 0.2822],
                 std=[0.2527, 0.1889, 0.1334],
                 size=(416, 416),
                 mode='train'):
        self.size = size
        self.mode = mode

        if self.mode == 'train':
            self.transforms = transformsCompose([
                ContrastEnhancement(factor=1.5),
                Resize(size),
                Fix_RandomRotation(),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                Normalize(mean, std)
            ])
        else:
            self.transforms = transformsCompose([
                ContrastEnhancement(factor=1.5),
                Resize(size),
                ToTensor(),
                Normalize(mean, std)
            ])

    def __call__(self, img, keypoints):
        img, keypoints = self.transforms(img, keypoints)
        return img, keypoints
    

    