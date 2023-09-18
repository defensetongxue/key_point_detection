import json,os
from utils_ import visualize_and_save_landmarks
from config import get_config
if __name__ =='__main__':
    args=get_config()
    with open(os.path.join('./split/1.json'),'r') as f:
        split_list=json.load(f)
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    for image_name in split_list['test']:
        data=data_dict[image_name]
        x,y=data['optic_disc_gt']['position']
        print(x,y)
        visualize_and_save_landmarks(data['image_path'],(1600,1200),(x,y),save_path=os.path.join('./experiments',image_name))
        raise