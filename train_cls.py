from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import get_config
import numpy as np
from utils_ import get_instance, train_epoch, val_epoch,get_optimizer,lr_sche
from Datasets_ import ClassDataset
from models import cls_models
import os
from utils_.function_ import to_device
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

# Create the model and criterion
model, criterion = cls_models(args.configs['model']['name'],args.configs['model'])

if os.path.isfile(args.from_checkpoint):
    print(f"loadding the exit checkpoints {args.from_checkpoint}")
    model.load_state_dict(
    torch.load(args.from_checkpoint))
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']
# Load the datasets
train_dataset = ClassDataset(args.data_path,args.configs,split_name=args.split_name,split="train")
val_dataset = ClassDataset(args.data_path,args.configs,split_name=args.split_name,split="val")
# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=args.configs['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'])
val_loader = DataLoader(val_dataset, batch_size=args.configs['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
print("train data number: ",len(train_dataset))
print("val data number: ",len(val_dataset))
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion=criterion.to(device)
print(f"using {device} for training")
# Set up the optimizer, loss function, and early stopping

early_stop_counter = 0
best_val_loss = float('inf')
total_epoches=args.configs['train']['end_epoch']
# Training and validation loop
for epoch in range(last_epoch,total_epoches):
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss = val_epoch(model, val_loader, criterion, device,epoch)
    
    print(f"Epoch {epoch + 1}/{total_epoches}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,f'{args.split_name}_{args.save_name}'))
        print("Model saved as {}".format(os.path.join(args.save_dir,f'{args.split_name}_{args.save_name}')))
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break
model.load_state_dict(torch.load(os.path.join(args.save_dir,f'{args.split_name}_{args.save_name}')))
test_split_name= args.split_name if len(args.split_name)>1 else 'u_'+args.split_name # using unvisual data for test
test_dataset= ClassDataset(args.data_path,args.configs,split_name=test_split_name,split="test")
test_loader=DataLoader(test_dataset, batch_size=args.configs['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
print(f"test data_number {len(test_dataset)}")
model.eval()
all_targets = []
all_predictions = []
all_probs = []

with torch.no_grad():
    for inputs, targets,meta in val_loader:
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)
        outputs = model(inputs)
        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=1)

        # Get predictions
        predictions = torch.argmax(probs, dim=1)

        # Store targets, predictions, and probabilities of positive class
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilities of class 1

# Convert lists to numpy arrays
all_targets = np.array(all_targets)
all_predictions = np.array(all_predictions)
all_probs = np.array(all_probs)

# Calculate accuracy and AUC
accuracy = accuracy_score(all_targets, all_predictions)
auc = roc_auc_score(all_targets, all_probs)

print("Accuracy:", accuracy)
print("AUC:", auc)