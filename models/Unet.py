from typing import Any
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(config.IN_CHANNELS, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)

        # Middle
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.classifier = nn.Sequential(
           nn.AdaptiveAvgPool2d((1, 1)),  # Pooling
           nn.Flatten(),  # Flatten the tensor before fully connected layers
           nn.Linear(1024, 512),  # Fully connected layer
           nn.ReLU(inplace=True),  # Activation
           nn.Dropout(0.5),  # Dropout for regularization
           nn.Linear(512,config.num_classes),  # Fully connected layer for num_classes output
        )
    def forward(self, x):
        # Encoder
        # x shape: [batch, in_channels, 256, 256]
        x1 = self.enc_conv1(x)  # [batch, 16, 256, 256]
        x2 = self.enc_conv2(x1)  # [batch, 32, 256, 256]
        pool1 = self.pool1(x2)  # [batch, 32, 128, 128]

        x3 = self.enc_conv3(pool1)  # [batch, 64, 128, 128]
        x4 = self.enc_conv4(x3)  # [batch, 128, 128, 128]
        pool2 = self.pool2(x4)  # [batch, 128, 64, 64]

        x5 = self.enc_conv5(pool2)  # [batch, 256, 64, 64]
        x6 = self.enc_conv6(x5)  # [batch, 512, 64, 64]
        pool3 = self.pool3(x6)  # [batch, 512, 32, 32]

        x7 = self.enc_conv7(pool3)  # [batch, 1024, 32, 32]
        pool4 = self.pool4(x7)  # [batch, 1024, 16, 16]

        # Middle
        mid = self.conv1(pool4)  # [batch, 1024, 16, 16]

        # Decoder
        up1 = self.upconv1(mid)  # [batch, 512, 32, 32]
        merge1 = torch.cat([up1, x7], dim=1)  # [batch, 1024, 32, 32]
        dec1 = self.dec_conv1(merge1)  # [batch, 512, 32, 32]

        up2 = self.upconv2(dec1)  # [batch, 256, 64, 64]
        merge2 = torch.cat([up2, x6], dim=1)  # [batch, 512, 64, 64]
        dec2 = self.dec_conv2(merge2)  # [batch, 256, 64, 64]

        up3 = self.upconv3(dec2)  # [batch, 128, 128, 128]
        merge3 = torch.cat([up3, x4], dim=1)  # [batch, 256, 128, 128]
        dec3 = self.dec_conv3(merge3)  # [batch, 128, 128, 128]

        out = self.out_conv(dec3)  # [batch, 1, 128, 128]
        
        return out,self.classifier(pool4)
class Loss_Unet():
    def __init__(self) -> None:
        self.location_loss=torch.nn.MSELoss()
        self.class_loss=torch.nn.CrossEntropyLoss()

    def __call__(self,input_heatmap ,target_heatmap,input_class,target_class ) :
        return self.location_loss(input_heatmap,target_heatmap)+0.2*self.class_loss(input_class,target_class)
def Build_UNet(config):
    
    model = UNet(config)
    pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    model.init_weights(pretrained=pretrained)

    return model,Loss_Unet()
if __name__ =="__main__":
    # Initialize the model
    model,citeria = Build_UNet(in_channels=1) 

    # Print the model architecture
    print(model)
