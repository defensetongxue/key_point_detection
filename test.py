import argparse
import os
import torch
from torch.utils.data import DataLoader
from config import get_config
from Datasets_ import get_keypoint_dataset
from utils_ import visualize_result,get_instance
import models

if __name__ == '__main__':
    args = get_config()
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    model, criterion = get_instance(models, args.model)
    model.load_state_dict(
        torch.load(os.path.join('./checkpoint',args.save_name)))
    model.to(device)
    model.eval()

    # Create the dataset and data loader
    data_path=os.path.join(args.path_tar, args.dataset)
    test_dataset = get_keypoint_dataset(data_path,split='test', output_format=model.output_format)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create the visualizations directory if it doesn't exist
    visual_dir = os.path.join(args.result_path, 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    # Test the model and save visualizations
    with torch.no_grad():
        for i, (inputs, targets,prescence) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            inputs=inputs.squeeze()
            targets=targets.squeeze()
            outputs=outputs[0].squeeze()
            target_path = os.path.join(visual_dir, f'{i}.jpg')
            visualize_result(inputs.cpu(), outputs.cpu(), targets, target_path, model.output_format)

    print("Finished testing")

