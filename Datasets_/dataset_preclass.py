import torch
import os,json
from torch.utils.data import Dataset
from PIL import Image,ImageEnhance
from torchvision import transforms
from torchvision.transforms import functional as F

class PreClassDataset(Dataset):
    def __init__(self, data_path,configs,split_name ,split='train'):
        self.data_path = data_path
        self.split=split
        # Load annotations
        with open(os.path.join(os.path.join(data_path,'annotations.json')),'r') as f:
            self.data_dict=json.load(f)
        with open(os.path.join('./split',f'{split_name}.json'),'r') as f:
            self.split_list=json.load(f)[split]
        print(f'using split {split_name}.json for  {split}')
        self.img_preprocess=transforms.Compose(
            [transforms.Resize(configs['image_resize']),
             ContrastEnhancement()])
        self.enhance = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
        self.label_map={"visible":0,'near':0,'far':1}

    def __len__(self):
        return len(self.split_list)
    
    def __getitem__(self, idx):
        image_name = self.split_list[idx]
        data=self.data_dict[image_name]

        img = Image.open(data['image_path']).convert('RGB')
        label=self.label_map[
            data['optic_disc_gt']['distance']
        ]
        # preprocess
        img=self.img_preprocess(img)
        if self.split=='train':
            img=self.enhance(img)

        img=self.img_transforms(img)
        assert label<2
        return img, label,image_name
        
class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img


class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
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

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, F.InterpolationMode.NEAREST , self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


