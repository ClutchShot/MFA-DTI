from torch import nn
import torch

class Conv(nn.Module):
    def __init__(self, args):
        super(Conv, self).__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(         
            nn.Conv2d(4, 2, 3, stride=2),                          
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 3, stride=1),                          
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((16, 16))   
        )


        self.conv1_old = nn.Sequential(         
            nn.Conv2d(1, 2, kernel_size=(5,5)),
            nn.BatchNorm2d(2),
            nn.MaxPool2d(5)
        )

        self.conv2_old = nn.Sequential(        
            nn.Conv2d(2, 4, kernel_size=(3,3)),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(3),
        )

        self.conv3 = nn.Sequential(        
            nn.Conv2d(4, 6, kernel_size=(1,1)),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2),
        )

        # self.layer1 = nn.Sequential(nn.Linear(4 * 37 * 7,440), nn.ReLU(True))
        self.layer1 = nn.Sequential(nn.Linear(4 * 67 * 7,512), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(512,256), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(256,128))


    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv1_old(x)
        x = self.conv2_old(x)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x    
    
