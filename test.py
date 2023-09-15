import argparse
import os
import torch
from torch.utils.data import DataLoader
from config import get_config
from Datasets_ import CustomDatset
from utils_ import decode_preds,visualize_and_save_landmarks,get_instance
import models

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
    torch.load(args.save_name))
model.eval()
# Create the dataset and data loader
test_dataset = CustomDatset(args.data_path,args.configs,split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# Create the visualizations directory if it doesn't exist

visual_dir = os.path.join(args.result_path, 'visual')
os.makedirs(visual_dir, exist_ok=True)
# Test the model and save visualizations
with torch.no_grad():
    for inputs, targets,meta in test_loader:
        inputs = inputs.to(device)

        output = model(inputs)[0].unsqueeze(0)
        score_map = output.data.cpu()
        preds = decode_preds(score_map)
        visualize_and_save_landmarks(
            image_path=meta[0],
            image_resize=args.configs['image_resize'],
            preds=preds,
            save_path=os.path.join(visual_dir,os.path.basename(meta[0])))
    print("Finished testing")

