
from torch import nn


class STblock(nn.Module):
    def __init__(self):
        super(STblock, self).__init__()

        self.STblock = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Conv3d(3, 3, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.Conv3d(3, 3, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Conv3d(3, 3, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1))
        )

    def forward(self, x):
        x = self.STblock(x)
        return x
