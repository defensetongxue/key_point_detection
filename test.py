import os,json
import torch
from config import get_config
from utils_ import decode_preds,visualize_and_save_landmarks,get_instance
import models
from torchvision import transforms
from PIL import Image
import numpy as np
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion

print("load the checkpoint in {}".format(os.path.join(args.save_dir,f'{args.split_name}_{args.save_name}')))
model, criterion = get_instance(models, args.configs['model']['name'],args.configs['model'],split='test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f'{args.split_name}_{args.save_name}')))
model.eval()

# Create the dataset and data loader

# Transform define
mytransforms = transforms.Compose([
            transforms.Resize(args.configs['image_resize']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)
        ])
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open(os.path.join('./split',f'{args.split_name}.json'),'r') as f:
    # split_ilst=json.load(f)['test'][:20]
    split_list=json.load(f)['test']
os.makedirs(os.path.join(args.result_path, 'visual'),exist_ok=True)
visual_dir = os.path.join(args.result_path, 'visual',args.split_name)
os.makedirs(visual_dir,exist_ok=True)
os.makedirs(os.path.join(visual_dir,'visual'),exist_ok=True)
os.makedirs(os.path.join(visual_dir,'unvisual'),exist_ok=True)
visual_list=[]
un_v=[]
mask_resize=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()])
cnt=0
test_list=['4605.jpg']
with torch.no_grad():
    # open the image and preprocess
    # for image_name in split_list:
    for image_name in test_list:
        data=data_dict[image_name]
        img=Image.open(data['enhanced_path']).convert('RGB')
        mask=Image.open(data['mask_path']).convert('L')
        mask=mask_resize(mask)
        mask[mask>0]=1
        
        ori_w,ori_h=img.size
        w_ratio,h_ratio=ori_w/args.configs['image_resize'][0], ori_h/args.configs['image_resize'][1]
        img = mytransforms(img)
        img = img.unsqueeze(0)  # as batch size 1
        position = model(img.cuda())
        score_map = position.data.cpu()
        score_map=score_map*mask
        # print(score_map.shape)
        preds = decode_preds(score_map)
        preds=preds.squeeze()
        preds=preds*np.array([w_ratio,h_ratio])
        max_val=torch.max(score_map)
        max_val=float(max_val)
        max_val=round(max_val,5)
        # if data['optic_disc_gt']['distance']=='visible':
        print(preds)
        if False:
            visual_list.append(max_val)
            visualize_and_save_landmarks(image_path=data['image_path'],
                                     preds=preds,
                                     save_path=os.path.join(visual_dir,'visual',image_name),
                                     text=max_val)
        else:
            cnt+=1
            un_v.append(max_val)
            # if max_val>=0.15:
            visualize_and_save_landmarks(image_path=data['image_path'],
                                     preds=preds,
                                    #  save_path=os.path.join(visual_dir,image_name),
                                     save_path=os.path.join(visual_dir,image_name),
                                     text=max_val)
print(cnt)
visual_list=sorted(visual_list)
un_v=sorted(un_v)
print(visual_list[:10])
print(un_v[-10:])