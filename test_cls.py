import torch
from torch.utils.data import DataLoader
from config import get_config
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
import numpy as np
from Datasets_ import ClassDataset
from models import cls_models
import os
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)

# Parse arguments
args = get_config()
print(f"using config file {args.cfg}")
# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device} for training")
# Create the model and criterion
model, criterion = cls_models(args.configs['model']['name'],args.configs['model'])
model = model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f'{args.configs["split_name"]}_{args.save_name}')))
model.eval()
# Load the datasets
test_dataset = ClassDataset(args.data_path,args.configs,split="test")
# Create the data loaders
test_loader = DataLoader(test_dataset, batch_size=args.configs['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
print("train data number: ",len(test_loader))
# List to store the true labels and the model's predictions
all_targets = []
all_outputs = []

# Testing loop
with torch.no_grad():
    for inputs, targets, meta in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        # Calculate loss if necessary (not needed for accuracy and AUC)
        l_loss, c_loss, loss = criterion(outputs, targets)

        # Apply softmax to get probabilities
        probas = F.softmax(outputs, dim=1)
        
        # Store the true labels and the predictions
        all_targets.append(targets.cpu().numpy())
        all_outputs.append(probas.cpu().numpy())

# Concatenate the results from each batch
all_targets = np.concatenate(all_targets)
all_outputs = np.concatenate(all_outputs)

# Calculate the accuracy and AUC
acc = accuracy_score(all_targets, np.argmax(all_outputs, axis=1))
auc = roc_auc_score(all_targets, all_outputs, multi_class='ovo')

print(f"Accuracy: {acc:.2f}")
print(f"AUC: {auc:.2f}")