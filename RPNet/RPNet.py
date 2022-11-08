from torch import nn
from .DilatedBlock import DilatedBlock

class RPNet(nn.Module):
    def __init__(self):
        super(RPNet, self).__init__()
        self.adaptpool = nn.AdaptiveAvgPool3d((150, 64, 64))
        self.Conv_1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.Conv_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.Dilated_Block_1 = DilatedBlock(64, 64, 1)
        self.Dilated_Block_2 = DilatedBlock(64, 64, 2)
        self.Dilated_Block_3 = DilatedBlock(64, 64, 4)

        self.DilConv = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(16, 1, 1), dilation=(8, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout3d(0.75)
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 1, 1))
        self.Conv_3 = nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def forward(self, x):
        x = self.adaptpool(x)
        x = self.Conv_1(x)
        x = self.maxpool(x)
        x = self.Conv_2(x)
        x = self.Dilated_Block_1(x)
        x = self.Dilated_Block_2(x)
        x = self.Dilated_Block_3(x)
        x = self.DilConv(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.Conv_3(x)
        x = x.view(150, 1)

        return x
