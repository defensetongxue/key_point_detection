import torch

def train_epoch(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(val_loader)

def test_epoch(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            raise NotImplementedError("visualize please")
            # predictions = extract_keypoints(outputs)  # You should implement the extract_keypoints function

            # all_predictions.extend(predictions)
            all_ground_truths.extend(targets.numpy())

    return all_predictions, all_ground_truths
