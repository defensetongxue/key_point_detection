import torch
from .visual_functional import decode_preds
import torch.nn.functional as F
import numpy as np
from PIL import Image
def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)

def train_epoch(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0
    for inputs, targets, meta in train_loader:
        # Moving inputs and targets to the correct device
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)
        
        # Assuming your loss function can handle tuples of outputs and targets
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # raise
    running_loss /= len(train_loader)

    running_loss=round(running_loss,8)
    return running_loss
def val_epoch(model, val_loader, loss_function, device,epoch):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets,meta in val_loader:
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)
            outputs = model(inputs)
            loss  = loss_function(outputs, targets)

            running_loss += loss.item()
        
    running_loss /= len(val_loader)
    running_loss=round(running_loss,8)
    return running_loss
def find_nearest_zero(mask, point):
    h, w = mask.shape
    cx, cy = w // 2, h // 2  # center of the image
    x, y = point

    if mask[y, x] == 0:
        return point

    # Get the direction vector for stepping along the line
    direction = np.array([x - cx, y - cy])
    direction = direction / np.linalg.norm(direction)

    # Step along the line until we hit a zero in the mask
    while 0 <= x < w and 0 <= y < h:
        if mask[int(y), int(x)] == 0:
            return (int(x), int(y))
        
        x += direction[0]
        y += direction[1]

    return (x,y)
