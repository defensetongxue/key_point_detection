import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import get_instance, train_epoch, val_epoch
from Datasets_ import get_keypoint_dataset
from torch import optim
import models
import os
# Parse arguments
args = get_config()

# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
if not os.path.exists(result_path):
    os.mkdir(result_path)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model, criterion = get_instance(models, args.model)
output_format = model.output_format

# Load the datasets
data_path=os.path.join(args.path_tar, args.dataset)
train_dataset = get_keypoint_dataset(data_path,"train", output_format)
val_dataset = get_keypoint_dataset(data_path,"valid", output_format)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")
# Set up the optimizer, loss function, and early stopping
optimizer = get_instance(optim, args.optimizer_type,
                         model.parameters(), lr=args.lr)

early_stop_counter = 0
best_val_loss = float('inf')

# Training and validation loop
for epoch in range(args.epoch):
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{args.epoch}, Train Loss: {train_loss}")

    val_loss = val_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{args.epoch}, Val Loss: {val_loss}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(),(os.path.join('./checkpoint',args.save_name)))
        print(f"Model saved as {args.save_name}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.early_stop:
            print("Early stopping triggered")
            break
