from config import get_config
import os,json
from api_record import api_update
from collections import defaultdict
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
def generate_split(data_list,split_train,split_val,split_file_path):
    '''
        generate a split json file in such format:
            {
                "train": [list for image_name],
                "val": [list for image name],
                "test: [list for image  name]
            }
        data list is in such format:
        {
            <image_name>: "class",
                            "fid":, 
        }
        split rule: 1. Try to make the sample proportions of several class consistent in train,val and test set.
                    2. the image in same fid should not be split into different sets
    '''
    # Step 1: Group data by label and then by fid
    grouped_data = defaultdict(lambda: defaultdict(list))

    for img_name, info in data_list.items():
        label = info['class']
        fid = info['fid']
        grouped_data[label][fid].append(img_name)

    # Step 2: Calculate splits
    train_data = []
    val_data = []
    test_data = []

    for label, fids in grouped_data.items():
        total_fids = len(fids)
        num_train = int(total_fids * split_train)
        num_val = int(total_fids * split_val)
        
        # Step 3: Create splits maintaining proportion of labels
        train_fids = list(fids.keys())[:num_train]
        val_fids = list(fids.keys())[num_train:num_train + num_val]
        test_fids = list(fids.keys())[num_train + num_val:]

        # Step 4: Add image names to the splits based on their fids
        train_data.extend([img for fid in train_fids for img in fids[fid]])
        val_data.extend([img for fid in val_fids for img in fids[fid]])
        test_data.extend([img for fid in test_fids for img in fids[fid]])

    # Create the final JSON structure
    split_json = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    # Write to a JSON file
    with open(split_file_path, 'w') as f:
        json.dump(split_json, f, indent=4)
print("All datasets processed successfully.")
