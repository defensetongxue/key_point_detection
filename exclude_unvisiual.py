import os,json
def exclude_uvisual(data_path,split_name):
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    with open(os.path.join('.','split',f'{split_name}.json'),'r') as f:
        ori_split=json.load(f)
    for split in ori_split:
        for image_name in ori_split[split]:
            data=data_dict[image_name]
            if data['optic_disc_gt']['distance']!="visible":
                ori_split[split].remove(image_name)
    with open(os.path.join('.','split',f'v_{split_name}.json'),'w') as f:
        json.dump(ori_split,f)
def exclude_visual(data_path,split_name):
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    with open(os.path.join('.','split',f'{split_name}.json'),'r') as f:
        ori_split=json.load(f)
    for split in ori_split:
        for image_name in ori_split[split]:
            data=data_dict[image_name]
            if data['optic_disc_gt']['distance']=="visible":
                ori_split[split].remove(image_name)
    with open(os.path.join('.','split',f'u_{split_name}.json'),'w') as f:
        json.dump(ori_split,f)
if __name__=='__main__':
    from config import get_config
    args=get_config()
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    with open(os.path.join('.','split',f'{args.split_name}.json'),'r') as f:
        ori_split=json.load(f)
    visual={}
    unvisual={}
    for split in ori_split:
        visual[split]=[]
        unvisual[split]=[]
        for image_name in ori_split[split]:
            data=data_dict[image_name]
            if data['optic_disc_gt']['distance']=="visible":
                visual[split].append(image_name)
            else:
                unvisual[split].append(image_name)
    with open(os.path.join('.','split',f'u_{args.split_name}.json'),'w') as f:
        json.dump(unvisual,f)
    with open(os.path.join('.','split',f'v_{args.split_name}.json'),'w') as f:
        json.dump(visual,f)