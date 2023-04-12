import argparse
import os
import torch
from torch.utils.data import DataLoader
from config import get_config
from Datasets_ import get_keypoint_dataset
from utils_ import visualize_result

if __name__ == '__main__':
    args = get_config()

    # Load the model
    model = torch.load(args.model_path)
    model.to(args.device)
    model.eval()

    # Create the dataset and data loader
    test_dataset = get_keypoint_dataset(split='test', output_format=model.output_format)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create the visualizations directory if it doesn't exist
    visual_dir = os.path.join(args.result_path, 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    # Test the model and save visualizations
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            outputs,targets=outputs.squeeze(),targets.squeeze()
            target_path = os.path.join(visual_dir, f'{i}.jpg')
            visualize_result(inputs.cpu(), outputs.cpu(), targets, target_path, model.output_format)

    print("Finished testing")

