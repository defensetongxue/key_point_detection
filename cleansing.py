import os,json
# Parse arguments
def generate_optic_disc_split(data_path,split_name):
    os.makedirs('./split',exist_ok=True)
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    with open(os.path.join(data_path,'split',f'{split_name}.json'),'r') as f:
        split_orignal=json.load(f)
    optic_disc_split={'train':[],'val':[],'test':[]}
    for split in ['train','val','test']:
        for image_name in split_orignal[split]:
            if 'optic_disc_gt' not in  data_dict[image_name]:
                continue
            optic_disc_split[split].append(image_name)
    with open(os.path.join('./split',f'{split_name}.json'),'w') as f:
        json.dump(optic_disc_split,f)
if __name__ =="__main__":
    from config import get_config
    args = get_config()
    