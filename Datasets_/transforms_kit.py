import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, heatmap):
        img = F.resize(img, self.size)
        heatmap = F.resize(heatmap, self.size, interpolation=Image.NEAREST)
        return img, heatmap

class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img, label):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img, label
     
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

    def __call__(self, img, heatmap):
        angle = self.get_params()
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        heatmap = F.rotate(heatmap, angle, resample=Image.NEAREST, expand=self.expand, center=self.center)
        return img, heatmap

class RandomHorizontalFlip:
    def __call__(self, img, heatmap):
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
            heatmap = F.hflip(heatmap)
        return img, heatmap

class RandomVerticalFlip:
    def __call__(self, img, heatmap):
        if torch.rand(1) < 0.5:
            img = F.vflip(img)
            heatmap = F.vflip(heatmap)
        return img, heatmap
