import os
import torch.nn as nn
from torchvision.models import resnet18

class Resnet18Regression(nn.Module):
    def __init__(self, num_keypoints,
                 download_path='./experiment/'):
        super(Resnet18Regression, self).__init__()
        os.environ['TORCH_HOME'] = download_path
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(self.resnet18.fc.in_features, num_keypoints * 2),
            nn.Tanh()
        )

    def forward(self, x):
        x= self.resnet18(x)
        x = (x + 1) / 2 # Scale and shift the output to the 0-1 range
        return x
    
def build_resnet18_regression(num_keypoints=1):
    model = Resnet18Regression(num_keypoints)
    model.output_format = "regression"
    loss_function = nn.MSELoss()
    return model, loss_function