import os
import torch.nn as nn
from torchvision.models import resnet18

class Resnet18Regression(nn.Module):
    def __init__(self, num_keypoints, download_path='./experiment/'):
        super(Resnet18Regression, self).__init__()
        os.environ['TORCH_HOME'] = download_path
        self.resnet18 = resnet18(pretrained=True)
        
        # Keypoint coordinates regression.
        self.coords_output = nn.Sequential(
            nn.Linear(self.resnet18.fc.in_features, num_keypoints * 2),
            nn.Tanh()
        )

        # Keypoint presence classification.
        self.presence_output = nn.Sequential(
            nn.Linear(self.resnet18.fc.in_features, 1),
            nn.Sigmoid()
        )

        # Remove original fully connected layer.
        self.resnet18.fc = nn.Identity()

    def forward(self, x):
        features = self.resnet18(x)
        coords = self.coords_output(features)
        presence = self.presence_output(features)

        # Scale and shift the coordinates output to the 0-1 range.
        coords = (coords + 1) / 2

        return coords, presence
def combined_loss(coords_pred, presence_pred, coords_gt, presence_gt):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    regression_loss = mse_loss(coords_pred, coords_gt)
    classification_loss = bce_loss(presence_pred, presence_gt)

    total_loss = regression_loss + classification_loss
    return total_loss
    
def build_resnet18_regression(num_keypoints=1):
    model = Resnet18Regression(num_keypoints)
    model.output_format = "regression"
    loss_function = combined_loss()
    return model, loss_function