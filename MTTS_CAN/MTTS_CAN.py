import torch
from torch import nn


def TSM(x):

    nt, c, w, h = x.size()
    x = x.reshape(-1, nt, c, w, h)
    fold = c // 3
    last_fold = c - (3 - 1) * fold
    out1, out2, out3 = torch.split(x, [fold, fold, last_fold], dim=2)

    # Shift left
    padding_1 = torch.zeros_like(out1)
    padding_1 = padding_1[:, 1, :, :, :]
    padding_1 = padding_1.unsqueeze(1)
    _, out1 = torch.split(out1, [1, nt - 1], dim=1)
    out1 = torch.cat([out1, padding_1], dim=1)

    # Shift right
    padding_2 = torch.zeros_like(out2)
    padding_2 = padding_2[:, 0, :, :, :]
    padding_2 = padding_2.unsqueeze(1)
    out2, _ = torch.split(out2, [nt - 1, 1], dim=1)
    out2 = torch.cat([padding_2, out2], dim=1)

    out = torch.cat([out1, out2, out3], dim=2)
    out = out.reshape(-1, c, w, h)

    return out


class MTTS_CAN_Model(nn.Module):
    def __init__(self):
        super(MTTS_CAN_Model, self).__init__()

        self.a_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn1 = nn.BatchNorm2d(32)
        self.a_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn2 = nn.BatchNorm2d(32)
        self.a_d1 = nn.Dropout2d(p=0.50)

        self.a_softconv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.a_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_d2 = nn.Dropout2d(p=0.25)
        self.a_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn3 = nn.BatchNorm2d(32)
        self.a_conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_bn4 = nn.BatchNorm2d(64)
        self.a_softconv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Motion stream
        self.m_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_bn1 = nn.BatchNorm2d(32)
        self.m_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_bn2 = nn.BatchNorm2d(32)

        self.m_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.d1 = nn.Dropout2d(p=0.25)

        self.m_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_bn3 = nn.BatchNorm2d(32)
        self.m_conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.m_bn4 = nn.BatchNorm2d(64)

        self.m_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.d2 = nn.Dropout2d(p=0.50)

        # Fully conected blocks
        self.d3 = nn.Dropout(p=0.50)

        self.fully1 = nn.Linear(in_features=64 * 9 * 9, out_features=128, bias=True)
        self.fully2 = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, A, M):
        n_frame = A.size(0)
        A = torch.mean(A, 0)
        A = A.repeat(n_frame, 1, 1, 1)
        A = torch.tanh(self.a_bn1(self.a_conv1(A)))
        A = torch.tanh(self.a_bn2(self.a_conv2(A)))

        # Calculating attention mask1 with softconv1
        mask1 = torch.sigmoid(self.a_softconv1(A))
        B, _, H, W = A.shape
        norm = 2 * torch.norm(mask1, p=1, dim=[1, 2, 3])
        norm = norm.reshape(B, 1, 1, 1)
        mask1 = torch.div(mask1 * H * W, norm)
        self.mask1 = mask1

        # Pooling
        A = self.a_avgpool(A)
        A = self.a_d2(A)
        # Last two convolution
        A = torch.tanh(self.a_bn3(self.a_conv3(A)))
        A = torch.tanh(self.a_bn4(self.a_conv4(A)))

        # Calculating attention mask2 with softconv2
        mask2 = torch.sigmoid(self.a_softconv2(A))
        B, _, H, W = A.shape
        norm = 2 * torch.norm(mask2, p=1, dim=[1, 2, 3])
        norm = norm.reshape(B, 1, 1, 1)
        mask2 = torch.div(mask2 * H * W, norm)
        self.mask2 = mask2

        # (M) - Motion storcheam --------------------------------------------------------------------
        M = TSM(M)
        M = torch.tanh(self.m_bn1(self.m_conv1(M)))
        M = TSM(M)
        M = torch.tanh(self.m_bn2(self.m_conv2(M)))
        M = torch.mul(M, mask1)
        # Pooling
        M = self.m_avgpool1(M)
        M = self.d1(M)
        # Last convs
        M = TSM(M)
        M = torch.tanh(self.m_bn3(self.m_conv3(M)))
        M = torch.tanh(self.m_bn4(self.m_conv4(M)))
        M = torch.mul(M, mask2)  #
        M = self.d2(M)
        M = self.m_avgpool2(M)

        # (F) - Fully connected part -------------------------------------------------------------
        # Flatten layer out
        out = torch.flatten(M, start_dim=1)
        out = self.d3(out)
        out = self.fully1(out)
        out = torch.tanh(out)
        out = self.fully2(out)

        return out