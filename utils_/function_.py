import torch
    
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
    running_l_loss=0.0
    running_c_loss=0.0
    for inputs, targets, meta in train_loader:
        # Moving inputs and targets to the correct device
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)

        # Assuming your loss function can handle tuples of outputs and targets
        l_loss,c_loss,loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_l_loss +=  l_loss.item()
        running_c_loss +=  c_loss.item()
    running_loss /= len(train_loader)
    running_l_loss /= len(train_loader)
    running_c_loss /= len(train_loader)

    running_c_loss=round(running_c_loss,4)
    running_l_loss=round(running_l_loss,4)
    running_loss=round(running_loss,4)
    return running_l_loss,running_c_loss,running_loss
def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    running_l_loss=0.0
    running_c_loss=0.0

    with torch.no_grad():
        for inputs, targets,meta in val_loader:
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)

            outputs = model(inputs)
            l_loss,c_loss,loss  = loss_function(outputs, targets)

            running_loss += loss.item()
            running_l_loss +=  l_loss.item()
            running_c_loss +=  c_loss.item()
    running_loss /= len(val_loader)
    running_l_loss /= len(val_loader)
    running_c_loss /= len(val_loader)
    running_c_loss=round(running_c_loss,4)
    running_l_loss=round(running_l_loss,4)
    running_loss=round(running_loss,4)
    return running_l_loss,running_c_loss,running_loss
