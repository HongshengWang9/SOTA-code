import torch
from torch.nn import Module
from torch import nn
from .AttentionBlock import AttentionBlock

class DeepPhys(Module):
    def __init__(self):
        super(DeepPhys, self).__init__()

        in_channels = 3
        out_channels = 32
        kernel_size = 3
        attention_mask1 = None
        attention_mask2 = None
        p = 0.50

        #  ########### AppearanceModel_2D  ################

        self.a_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.a_batch_Normalization1 = nn.BatchNorm2d(out_channels)
        self.a_conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                       padding=1)
        self.a_batch_Normalization2 = nn.BatchNorm2d(out_channels)
        self.a_dropout1 = nn.Dropout2d(p=p)
        # Attention mask1
        self.attention_mask1 = AttentionBlock(out_channels)
        self.a_avg1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1,
                                       padding=1)
        self.a_Batch_Normalization3 = nn.BatchNorm2d(out_channels * 2)
        self.a_conv4 = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2, kernel_size=3,
                                       stride=1, padding=1)
        self.a_Batch_Normalization4 = nn.BatchNorm2d(out_channels * 2)
        self.a_dropout2 = nn.Dropout2d(p=p)
        # Attention mask2
        self.attention_mask2 = AttentionBlock(out_channels * 2)

        # ###################### MotionModel ######################

        self.m_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.m_batch_Normalization1 = nn.BatchNorm2d(out_channels)
        self.m_conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.m_batch_Normalization2 = nn.BatchNorm2d(out_channels)
        self.m_dropout1 = nn.Dropout2d(p=p)

        self.m_avg1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=kernel_size,
                                       stride=1,
                                       padding=1)
        self.m_batch_Normalization3 = nn.BatchNorm2d(out_channels * 2)
        self.m_conv4 = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                       kernel_size=kernel_size, stride=1, padding=1)
        self.m_batch_Normalization4 = nn.BatchNorm2d(out_channels * 2)
        self.m_dropout2 = nn.Dropout2d(p=p)
        self.m_avg2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # #################### LinearModel ###########################

        self.f_drop1 = nn.Dropout(0.25)
        self.f_linear1 = nn.Linear(in_features=64 * 9 * 9, out_features=128, bias=True)
        self.f_linear2 = nn.Linear(in_features=128, out_features=1, bias=True)
        # self.f_linear3 = nn.Linear(in_features=64 * 9 * 9, out_features=1, bias=True)

    def forward(self, input_1, input_2):

        #  ########### AppearanceModel_2D  ################
        # Convolution layer
        A1 = self.a_conv1(input_1)
        A1 = self.a_batch_Normalization1(A1)
        A1 = torch.tanh(A1)
        A2 = self.a_conv2(A1)
        A2 = self.a_batch_Normalization2(A2)
        A2 = torch.tanh(A2)
        A3 = self.a_dropout1(A2)
        # Calculate Mask1
        A_M1 = self.attention_mask1(A3)
        # Pooling
        A4 = self.a_avg1(A3)
        # Convolution layer
        A5 = self.a_conv3(A4)
        A5 = self.a_Batch_Normalization3(A5)
        A5 = torch.tanh(A5)
        A6 = self.a_conv4(A5)
        A6 = self.a_Batch_Normalization4(A6)
        A6 = torch.tanh(A6)
        A7 = self.a_dropout2(A6)
        # Calculate Mask2
        A_M2 = self.attention_mask2(A7)

        # ###################### MotionModel ######################

        M1 = self.m_conv1(input_2)
        M1 = self.m_batch_Normalization1(M1)
        M1 = torch.tanh(M1)
        M2 = self.m_conv2(M1)
        M2 = self.m_batch_Normalization2(M2)
        # element wise multiplication Mask1
        # ones = torch.ones(size=M2.shape).to('cuda')
        g1 = torch.tanh(torch.mul(M2, A_M1))
        M3 = self.m_dropout1(g1)
        # pooling
        M4 = self.m_avg1(M3)
        # g1 = torch.tanh(torch.mul(1 * mask1, M4))
        M5 = self.m_conv3(M4)
        M5 = self.m_batch_Normalization3(M5)
        M5 = torch.tanh(M5)
        M6 = self.m_conv4(M5)
        M6 = self.m_batch_Normalization4(M6)
        # element wise multiplication Mask2
        g2 = torch.tanh(torch.mul(M6, A_M2))
        M7 = self.m_dropout2(g2)
        M8 = self.m_avg2(M7)
        # out = torch.tanh(M8)
        out = M8

        # #################### LinearModel ###########################

        f1 = torch.flatten(out, start_dim=1)
        f2 = self.f_drop1(f1)
        f3 = torch.tanh(self.f_linear1(f2))
        f4 = self.f_linear2(f3)

        return f4