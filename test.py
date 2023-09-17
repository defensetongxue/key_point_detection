import os,json
import torch
from config import get_config
from utils_ import decode_preds,visualize_and_save_landmarks,get_instance
import models
from torchvision import transforms
from Datasets_ import ContrastEnhancement
from PIL import Image
import numpy as np
# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model, criterion = get_instance(models, args.model,args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f"{args.split_name}_{args.save_name}")))
print("load the checkpoint in {}".format(os.path.join(args.save_dir,f"{args.split_name}_{args.save_name}")))
model.eval()
# Create the dataset and data loader

# Transform define
transforms = transforms.Compose([
            ContrastEnhancement(),
            transforms.Resize(args.configs['image_resize']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])
        ])
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open(os.path.join('./split',f'{args.split_name}.json'),'r') as f:
    split_ilst=json.load(f)['test']
distance_map={0:"visible",1:"near",2:"far"}
visual_dir = os.path.join(args.result_path, 'visual')
with torch.no_grad():
    # open the image and preprocess
    for image_name in split_ilst:
        data=data_dict[image_name]
        img=Image.open(data[image_name]).convert('RGB')
        ori_w,ori_h=img.size
        w_ratio,h_ratio=ori_w/args.configs['image_resize'][0], ori_h/args.configs['image_resize'][1]
        img = transforms(img)
        # generate predic heatmap with pretrained   model
        img = img.unsqueeze(0)  # as batch size 1
        position,distance = model(img.cuda())
        # the input of the 512 is to match the  mini-size of vessel model
        score_map = position.data.cpu().unsqueeze(0)
        preds = decode_preds(score_map)
        preds=preds.squeeze()
        preds=preds*np.array([w_ratio,h_ratio])
        distance=torch.argmax(distance, dim=1).squeeze()
        distance=distance_map[int(distance)]
        visualize_and_save_landmarks(image_path=data['image_path'],
                                     image_resize=args.configs['image_resize'],
                                     perds=preds,
                                     save_path=os.path.join(visual_dir,image_name),
                                     text=distance)

