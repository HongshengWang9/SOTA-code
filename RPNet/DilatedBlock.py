from torch import nn


class DilatedBlock(nn.Module):
    def __init__(self, in_C, out_C, d):
        super(DilatedBlock, self).__init__()

        self.DilConv = nn.Sequential(
            nn.Conv3d(in_C, out_C, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2*d, 1, 1), dilation=(d, 1, 1)),
            nn.BatchNorm3d(out_C),
            nn.ReLU()
        )
        self.dropout = nn.Dropout3d(0.5)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.Conv = nn.Sequential(
            nn.Conv3d(out_C, out_C, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(out_C),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.DilConv(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.Conv(x)
        return x
