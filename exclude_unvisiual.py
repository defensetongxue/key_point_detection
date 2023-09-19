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
if __name__=='__main__':
    from config import get_config
    args=get_config()
    exclude_uvisual(args.data_path,args.configs['split_name'])