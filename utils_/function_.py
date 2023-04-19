import torch

def train_epoch(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0

    for inputs, targets ,presence in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, targets,presence)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets,presence in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets,presence)

            running_loss += loss.item()

    return running_loss / len(val_loader)
