import torch.nn as nn
class WangNet(nn.Module):
    def __init__(self):
        super(WangNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128, affine=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256, affine=True),
                                    nn.MaxPool2d(2, 2)

                                    )
        self.block4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512, affine=True),
                                    nn.MaxPool2d(kernel_size=2, dilation=2)
                                    )
        self.block5 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)

        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x1 = x.view(-1, 512 * 7 * 7)
        x1 = self.block5(x1)
        return x1, x