import os,json
from PIL import Image
import numpy as np
# Parse arguments
def generate_optic_disc_split(data_path,split_name):
    os.makedirs('./split',exist_ok=True)
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    with open(os.path.join(data_path,'split',f'{split_name[-1]}.json'),'r') as f:
        split_orignal=json.load(f)
    optic_disc_split={'train':[],'val':[],'test':[]}
    for split in ['train','val','test']:
        for image_name in split_orignal[split]:
            if 'optic_disc_gt' not in  data_dict[image_name]:
                continue
            optic_disc_split[split].append(image_name)
    with open(os.path.join('./split',f'{split_name}.json'),'w') as f:
        json.dump(optic_disc_split,f)

def generate_optic_disc_gt(data_path,sigma,heatmap_ratio,mask_path):
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    os.makedirs(os.path.join(data_path,'optic_disc_heatmap'),exist_ok=True)
    mask=Image.open(mask_path)
    # Get the dimensions of the image
    width, height = mask.size

    # Calculate the new dimensions
    new_width = int(width *heatmap_ratio)
    new_height = int(height * heatmap_ratio)

    # Resize the image
    mask= mask.resize((new_width, new_height))
    mask=np.array(mask,dtype=np.float16)

    for image_name in data_dict:
        data=data_dict[image_name]
        if 'optic_disc_gt' in data:
            optic_disc_heatmap_path=os.path.join(data_path,'optic_disc_heatmap',data['id']+'.png')
            data_dict[image_name]['optic_disc_heatmap_path']=generate_optic_disc_heatmap(
                optic_disc_gt= data['optic_disc_gt'],
                heatmap_ratio=heatmap_ratio,
                sigma=sigma,
                mask = mask,
                save_path=optic_disc_heatmap_path)
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_dict,f)
def generate_optic_disc_heatmap(optic_disc_gt,heatmap_ratio,sigma,mask,save_path):
    # build empty heatmap
    heatmap=np.zeros_like(mask)
    heatmap_punish=1.0
    pt=[i*heatmap_ratio for i in optic_disc_gt['position']]
    if optic_disc_gt['distance']=='far':
        sigma=sigma*3
        heatmap_punish=0.5
    elif optic_disc_gt['distance']=='near':
        sigma=sigma*2
        heatmap_punish=0.8
    
    # build heatmap
    tmp_size = sigma * 3
    ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
    br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
    
    # Adjust the bounds to fit within the image dimensions
    ul[0] = max(0, ul[0])
    ul[1] = max(0, ul[1])
    br[0] = min(heatmap.shape[1], br[0])
    br[1] = min(heatmap.shape[0], br[1])
    
    # Generate Gaussian
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable Gaussian range
    g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
    
    # Image range
    heatmap_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
    heatmap_y = max(0, ul[1]), min(br[1], heatmap.shape[0])
    
    heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    # mask and save
    heatmap=np.where(mask>1, heatmap* heatmap_punish,np.zeros_like(heatmap)) 
    heatmap=(heatmap*255).astype(np.uint8)
    img = Image.fromarray(heatmap)
    img.save(save_path)
    return save_path
if __name__ =="__main__":
    from config import get_config
    args = get_config()
    generate_optic_disc_split(args.data_path,args.split_name)
    # generate_optic_disc_gt(args.data_path,
    #                        sigma=args.configs['sigma'],
    #                        heatmap_ratio=args.configs['heatmap_rate'],
    #                        mask_path='./mask.png')