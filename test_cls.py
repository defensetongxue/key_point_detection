import torch
from torch.utils.data import DataLoader
from config import get_config
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix
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
    torch.load(os.path.join(args.save_dir,f'{args.split_name}_{args.save_name}')))
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
# ... other parts of the code ...

# Testing loop
with torch.no_grad():
    for inputs, targets, meta in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        
        # Apply softmax to get probabilities
        probas = F.softmax(outputs, dim=1)
        
        # Get the index of the max probability, which represents the predicted label
        _, predicted = torch.max(probas, 1)
        
        # Convert tensor to list for processing
        targets_list = targets.cpu().numpy().tolist()
        predicted_list = predicted.cpu().numpy().tolist()
        meta_list = list(meta)  # assuming meta is a tuple, convert it to a list

        # Iterate over the batch and print the incorrect predictions
        for i in range(len(targets_list)):
            true_label = targets_list[i]
            predicted_label = predicted_list[i]
            
            # If the prediction is wrong, print the image_name, true label, and predicted label
            if true_label != predicted_label:
                print(f"{meta_list[i]} {true_label} {predicted_label}")

        # Store the true labels and the predictions for further analysis
        all_targets.append(targets.cpu().numpy())
        all_outputs.append(probas.cpu().numpy())


# Concatenate the results from each batch
predicted_labels = np.concatenate([np.argmax(probas, axis=1) for probas in all_outputs])
all_outputs = np.concatenate(all_outputs)
true_labels = np.concatenate(all_targets)

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# If you want to print in "label i, pred j" format:
print("\nDetailed View:")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        print(f"label {i}, pred {j}: {cm[i, j]}")


# Calculate the accuracy and AUC
acc = accuracy_score(true_labels, np.argmax(all_outputs, axis=1))
auc = roc_auc_score(true_labels, all_outputs, multi_class='ovo')
