from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F
# class UNet(nn.Module):
#     def __init__(self, config):
#         super(UNet, self).__init__()

#         # Encoder
#         self.enc_conv1 = nn.Conv2d(config['in_channels'], 16, kernel_size=3, padding=1)
#         self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool2d(2)

#         self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool2d(2)

#         self.enc_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.enc_conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.pool3 = nn.MaxPool2d(2)

#         self.enc_conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
#         self.pool4 = nn.MaxPool2d(2)

#         # Middle
#         self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

#         # Decoder
#         self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.dec_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

#         self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

#         self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

#         self.out_conv = nn.Conv2d(128, 1, kernel_size=1)
#         self.classifier = nn.Sequential(
#            nn.AdaptiveAvgPool2d((1, 1)),  # Pooling
#            nn.Flatten(),  # Flatten the tensor before fully connected layers
#            nn.Linear(1024, 512),  # Fully connected layer
#            nn.ReLU(inplace=True),  # Activation
#            nn.Dropout(0.5),  # Dropout for regularization
#            nn.Linear(512,config['num_classes']),  # Fully connected layer for num_classes output
#         )
#     def forward(self, x):
#         # Encoder
#         # x shape: [batch, in_channels, 256, 256]
#         x1 = self.enc_conv1(x)  # [batch, 16, 256, 256]
#         x2 = self.enc_conv2(x1)  # [batch, 32, 256, 256]
#         pool1 = self.pool1(x2)  # [batch, 32, 128, 128]

#         x3 = self.enc_conv3(pool1)  # [batch, 64, 128, 128]
#         x4 = self.enc_conv4(x3)  # [batch, 128, 128, 128]
#         pool2 = self.pool2(x4)  # [batch, 128, 64, 64]

#         x5 = self.enc_conv5(pool2)  # [batch, 256, 64, 64]
#         x6 = self.enc_conv6(x5)  # [batch, 512, 64, 64]
#         pool3 = self.pool3(x6)  # [batch, 512, 32, 32]

#         x7 = self.enc_conv7(pool3)  # [batch, 1024, 32, 32]
#         pool4 = self.pool4(x7)  # [batch, 1024, 16, 16]

#         # Middle
#         mid = self.conv1(pool4)  # [batch, 1024, 16, 16]

#         # Decoder
#         up1 = self.upconv1(mid)  # [batch, 512, 32, 32]
#         merge1 = torch.cat([up1, x6], dim=1)  # [batch, 1024, 32, 32]
#         dec1 = self.dec_conv1(merge1)  # [batch, 512, 32, 32]

#         up2 = self.upconv2(dec1)  # [batch, 256, 64, 64]
#         merge2 = torch.cat([up2, x5], dim=1)  # [batch, 512, 64, 64]
#         dec2 = self.dec_conv2(merge2)  # [batch, 256, 64, 64]

#         up3 = self.upconv3(dec2)  # [batch, 128, 128, 128]
#         merge3 = torch.cat([up3, x4], dim=1)  # [batch, 256, 128, 128]
#         dec3 = self.dec_conv3(merge3)  # [batch, 128, 128, 128]

#         out = self.out_conv(dec3)  # [batch, 1, 128, 128]

#         return out,self.classifier(pool4)


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)


class UNet(nn.Module):

    def __init__(self, configs):
        super(UNet, self).__init__()

        self.C_embed = Conv(configs["in_channels"], 16)
        self.D_embed = DownSampling(16)
        self.C0 = Conv(16, 32)
        self.D0 = DownSampling(32)
        self.C1 = Conv(32, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Pooling
            nn.Flatten(),  # Flatten the tensor before fully connected layers
            nn.Linear(1024, 512),  # Fully connected layer
            nn.ReLU(inplace=True),  # Activation
            nn.Dropout(0.5),  # Dropout for regularization
            # Fully connected layer for num_classes output
            nn.Linear(512, configs['num_classes']),
        )

    def forward(self, x):
        embed = self.C_embed(x)
        R0 = self.C0(self.D_embed(embed))
        R1 = self.C1(self.D0(R0))
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))  # 1024

        O1 = self.C6(self.U1(Y1, R4))  # 512
        O2 = self.C7(self.U2(O1, R3))  # 256
        O3 = self.C8(self.U3(O2, R2))  # 128
        O4 = self.C9(self.U4(O3, R1))  # 64

        heatmap = self.Th(self.pred(O4)).squeeze()  # segment
        distance = self.classifier(Y1)
        return (heatmap, distance)


class Loss_Unet():
    def __init__(self, locat_r=0.7):
        self.r = locat_r
        # DiceLoss,FocalLoss
        self.location_loss = nn.BCELoss()
        self.class_loss = nn.CrossEntropyLoss()

    def __call__(self, ouputs, targets):
        out_heatmap, out_distance = ouputs
        gt_heatmap, gt_distance = targets
        return self.r*self.class_loss(out_heatmap, gt_heatmap) + \
            (1-self.r)*self.class_loss(out_distance, gt_distance)

def Build_UNet(config):

    model = UNet(config)
    # pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    # model.init_weights(pretrained=pretrained)

    return model, Loss_Unet()


if __name__ == "__main__":
    # Initialize the model
    model, citeria = Build_UNet(in_channels=1)

    # Print the model architecture
    print(model)
