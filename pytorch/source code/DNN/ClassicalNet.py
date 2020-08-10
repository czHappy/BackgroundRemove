import  torch
import  torch.nn as nn
import  torchvision
import numpy

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.conv_block = nn.Sequential( # input 227 227
            # A sequential container. Modules will be added to it in the order they are passed in the constructor.
            # Alternatively, an ordered dict of modules can also be passed in.
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3, stride=2),


            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3, stride=2)

        )

        self.classfier_block = nn.Sequential(
            nn.Dropout(p=0.5), #p: probability of an element to be zeroed
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(True),

            nn.Dropout(p=0.5),  # p: probability of an element to be zeroed
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(True),

            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), 6*6*256)
        x = self.classfier_block(x)
        return x


