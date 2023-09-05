from config import get_config
import os,json
from api_record import api_update
# Parse arguments
args = get_config()
def generate_optic_disc(json_dict,data_path):
    print(f"begin paser ridge from {json_dict}")
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        original_annoation=json.load(f)
    file_list=sorted(os.listdir(json_dict))
    print(f"read the origianl json file from {file_list}")
    for file in file_list:
        if not file.split('.')[-1]=='json':
            print(f"unexpected file {file} in json_src")
            continue
        with open(os.path.join(json,file), 'r') as f:
            data = json.load(f)
        for json_obj in data:
            image_name,new_data=parse_json(json_obj)
            original_annoation[image_name]['optic_disc_gt']=new_data
    api_update(data_path,'optic_disc_gt',"Location for optic_disc_gt")
    print("finished")
    
def parse_json(input_data):
    annotations = input_data.get("annotations", [])
    if annotations:
        result = annotations[0].get("result", [])
    image_name=input_data["file_upload"].split('-')[-1]
    new_data = {
        "distance": None,
        "position":None
    }

    for item in result:
        if item["type"] == "keypointlabels":
            x= item["value"]["x"]*item["original_width"]/100
            y= item["value"]["y"]*item["original_height"]/100
            label = item["value"]["keypointlabels"][0]

            if label == "O_D_u_99":
                new_data["distance"]= "far"
                new_data["position"]=(x, y)
            elif label == "O_D_u_v":
                new_data["distance"] = "visible"
                new_data["position"]=(x, y)
            elif label == "O_D_u_1":
                new_data["distance"] = "near"
                new_data["position"]=(x, y)

    return image_name,new_data

print("All datasets processed successfully.")
