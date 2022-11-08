import torch
from torch import nn
from .STblock import STblock


class ETA_rPPGNet2(nn.Module):
    def __init__(self, N, W, H):
        super(ETA_rPPGNet2, self).__init__()

        self.adapt = nn.AdaptiveAvgPool3d((3, 128, 128))
        # ###########  TD_segment_net ###########
        self.DW = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), groups=3),
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )

        self.avg_pool = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.PW = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )

        self.adapt_avgpool = nn.AdaptiveAvgPool3d((1, int(W / 2), int(H / 2)))
        # ###########  STblock  ###########
        self.STblock_1 = STblock()
        self.STblock_2 = STblock()
        self.STblock_3 = STblock()
        self.STblock_4 = STblock()
        # ###########  TD_Attention  ###########
        self.GAP = nn.AvgPool3d(kernel_size=(1, int(W/2 - 4), int(H/2 - 4)))
        self.oneconv = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )
        # #######################################
        self.adptavgpool = nn.AdaptiveAvgPool3d((N, 1, 1))
        self.Conv = nn.Conv3d(3, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def forward(self, x):
        x = self.adapt(x)
        # ###########  TD_segment_net ###########
        s = self.DW(x)
        s = self.avg_pool(s)
        s = self.PW(s)
        A = torch.sigmoid(s)
        s = x * (1 + A)
        x = self.adapt_avgpool(s)
        x = x.permute(2, 1, 0, 3, 4)
        # ###########  STblock  ###########
        x = self.STblock_1(x)
        x = self.STblock_2(x)
        xp = self.STblock_3(x)

        x = self.STblock_4(xp)
        s = self.GAP(x)
        s = self.oneconv(s)
        M = torch.sigmoid(s)
        M = M.repeat(1, 1, 1, x.size(3), x.size(4))
        x = x * (1 + M)

        x = self.adptavgpool(x)
        x = self.Conv(x)
        x = x.view(1, 1, x.size(2))
        x = nn.functional.interpolate(x, size=150, mode="linear", align_corners=True)
        x_pre = x.view(x.size(2), 1)

        s = self.GAP(xp)
        s = self.oneconv(s)
        M = torch.sigmoid(s)
        M = M.repeat(1, 1, 1, xp.size(3), xp.size(4))
        xp = xp * (1 + M)

        xp = self.adptavgpool(xp)
        xp = self.Conv(xp)
        xp = xp.view(1, 1, xp.size(2))
        xp = nn.functional.interpolate(xp, size=150, mode="linear", align_corners=True)
        x_post = xp.view(xp.size(2), 1)

        return x_pre, x_post