import os,json
import torch
from config import get_config
from utils_ import decode_preds,visualize_and_save_landmarks,get_instance,get_criteria
import models
from torchvision import transforms
from PIL import Image,ImageEnhance
import numpy as np
import torch.nn.functional as F
class ContrastEnhancement:
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.factor)
        return img
     
# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model, criterion = get_instance(models, args.configs['model']['name'],args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f'{args.configs["split_name"]}_{args.save_name}')))
print("load the checkpoint in {}".format(os.path.join(args.save_dir,f'{args.configs["split_name"]}_{args.save_name}')))
model.eval()
# Create the dataset and data loader

# Transform define
mytransforms = transforms.Compose([
            ContrastEnhancement(),
            transforms.Resize(args.configs['image_resize']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])
        ])
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open(os.path.join('./split',f'{args.configs["split_name"]}.json'),'r') as f:
    split_ilst=json.load(f)['test'][:20]
os.makedirs(os.path.join(args.result_path, 'visual'),exist_ok=True)
visual_dir = os.path.join(args.result_path, 'visual',args.configs["split_name"])
os.makedirs(visual_dir,exist_ok=True)
visual_list=[]
un_v=[]
with torch.no_grad():
    # open the image and preprocess
    for image_name in split_ilst:
        data=data_dict[image_name]
        img=Image.open(data['image_path']).convert('RGB')
        ori_w,ori_h=img.size
        w_ratio,h_ratio=ori_w/args.configs['image_resize'][0], ori_h/args.configs['image_resize'][1]
        img = mytransforms(img)
        img = img.unsqueeze(0)  # as batch size 1
        position = model(img.cuda())
        score_map = position.data.cpu()
        # print(score_map.shape)
        preds = decode_preds(score_map)
        preds=preds.squeeze()
        preds=preds*np.array([w_ratio,h_ratio])
        visualize_and_save_landmarks(image_path=data['image_path'],
                                     preds=preds,
                                     save_path=os.path.join(visual_dir,image_name))
        max_val=torch.max(score_map)
        max_val=float(max_val)
        max_val=round(max_val,5)

        if data['optic_disc_gt']['distance']=='visible':
            visual_list.append(max_val)
        else:
            un_v.append(max_val)
visual_list=sorted(visual_list)
un_v=sorted(un_v)
print(visual_list[:10])
print(un_v[-10:])