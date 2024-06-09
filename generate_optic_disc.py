import os,json
import torch
from config import get_config
from utils_ import decode_preds,visualize_and_save_landmarks,get_instance
import models
from torchvision import transforms
from PIL import Image
import numpy as np
from models import cls_models
from utils_ import find_nearest_zero
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Parse arguments
args = get_config()

model_v, _ = get_instance(models, args.configs['model']['name'],args.configs['model'],split='test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_v=model_v.to(device)
model_v.load_state_dict(
    torch.load(os.path.join(args.save_dir,f'v_{args.split_name}_{args.save_name}')))
model_v.eval()

model_u, _ = get_instance(models, args.configs['model']['name'],args.configs['model'],split='test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_u=model_u.to(device)
model_u.load_state_dict(
    torch.load(os.path.join(args.save_dir,f'u_{args.split_name}_{args.save_name}')))
model_u.eval()

model_cls, _  = cls_models(args.configs['cls_model']['name'],args.configs['cls_model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cls=model_cls.to(device)
model_cls.load_state_dict(torch.load(os.path.join(args.save_dir,f'{args.split_name}_optic_disc_cls.pth')))
model_cls.eval()
# Create the dataset and data loader

# Transform define
mytransforms = transforms.Compose([
            transforms.Resize(args.configs['image_resize']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)
        ])

cls_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
distance_map={
    0:'near',
    1:'far'
}
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open(os.path.join(args.data_path,'split',f'{args.split_name}.json'),'r') as f:
    split_list=json.load(f)['test']
    
mask_resize=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()])
cnt=0
with torch.no_grad():
    # open the image and preprocess
    for image_name in split_list:
        cnt+=1
        data=data_dict[image_name]
        img=Image.open(data['enhanced_path']).convert('RGB')
        mask=Image.open(data['mask_path']).convert('L')
        mask_tensor=mask_resize(mask)
        mask_tensor[mask_tensor>0]=1
        
        # pre
        ori_w,ori_h=img.size
        w_ratio,h_ratio=ori_w/args.configs['image_resize'][0], ori_h/args.configs['image_resize'][1]
        img = mytransforms(img)
        img = img.unsqueeze(0)  # as batch size 1
        position = model_v(img.cuda())
        score_map = position.data.cpu()
        score_map=score_map*mask_tensor
        preds = decode_preds(score_map)
        preds=preds.squeeze()
        preds=preds*np.array([w_ratio,h_ratio])
        max_val=torch.max(score_map)
        max_val=float(max_val)
        max_val=round(max_val,5)
        x,y = int(preds[0]),int(preds[1])
        distance='visible' 
        if max_val<args.threshold:
            # the optic disc is not detected succussfully
            position = model_u(img.cuda())
            score_map = position.data.cpu()
            score_map=score_map*mask_tensor
            preds = decode_preds(score_map)
            preds=preds.squeeze()
            preds=preds*np.array([w_ratio,h_ratio])
            
            mask=transforms.ToTensor()(mask)
            mask[mask>0]=1
            x,y=find_nearest_zero(mask.squeeze(),(int(preds[0]),int(preds[1])))
            
            cls_img=Image.open(data['image_path']).convert('RGB')
            cls_img=cls_transforms(cls_img).unsqueeze(0)
            outputs = model_cls(cls_img.cuda()).cpu()
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            predictions = torch.argmax(probs, dim=1)
            distance=distance_map[int(predictions)]
            
        data_dict[image_name]['optic_disc_pred']={
            'position':[int(x),int(y)],
            "visible_confidnce":max_val,
            'distance':distance}
print(f"finish process {str(cnt)} images")
with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)