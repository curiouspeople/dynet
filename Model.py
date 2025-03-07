import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


# class MultiHeadSelfAttention1(nn.Module):
#     dim_in: int  # input dimension
#     dim_k: int   # key and query dimension
#     dim_v: int   # value dimension
#     num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
#
#     def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
#         super(MultiHeadSelfAttention1, self).__init__()
#         assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
#         self.dim_in = dim_in
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         self.num_heads = num_heads
#         self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
#         self._norm_fact = 1 / sqrt(dim_k // num_heads)
#
#     def forward(self, x):
#         # x: tensor of shape (batch, c1, c2, dim_in)
#         batch, c1, c2, dim_in = x.shape
#         assert dim_in == self.dim_in
#
#         nh = self.num_heads
#         dk = self.dim_k // nh  # dim_k of each head
#         dv = self.dim_v // nh  # dim_v of each head
#
#         q = self.linear_q(x).reshape(batch, c1, c2, nh, dk).transpose(2, 3)  # (batch, nh, n, dk)
#         k = self.linear_k(x).reshape(batch, c1, c2, nh, dk).transpose(2, 3)  # (batch, nh, n, dk)
#         v = self.linear_v(x).reshape(batch, c1, c2, nh, dv).transpose(2, 3)  # (batch, nh, n, dv)
#
#         dist = torch.matmul(q, k.transpose(3, 4)) * self._norm_fact  # batch, nh, n, n
#         dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
#
#         att = torch.matmul(dist, v)  # batch, nh, n, dv
#         att = att.transpose(2, 3).reshape(batch, c1, c2, self.dim_v)  # batch, c1, c2, dim_v
#         return att


class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, c1, c2, dim_in)
        batch, c, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, c, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, c, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, c, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, c, self.dim_v)  # batch, c, dim_v
        return att


class Attentionadj(nn.Module):
    def __init__(self, in_size=64, hidden_size=64):
        super(Attentionadj, self).__init__()

        self.project = nn.Sequential(
            # nn.BatchNorm2d(3),
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.sigmoid(w)
        return beta * z


class LPE(nn.Module):
    def __init__(self):
        super(LPE, self).__init__()

        self.fc1 = nn.Linear(10, 1)
        self.multi_att1 = MultiHeadSelfAttention(64, 64, 64)
        self.bn = nn.BatchNorm1d(64)
        self.adaverage_pool = nn.AdaptiveAvgPool2d(output_size=(5, 1))

    def forward(self, vector):
        vector = self.fc1(vector.permute(0, 2, 3, 1, 4).reshape(vector.shape[0], vector.shape[2], vector.shape[3],
                                                                -1))  # torch.Size([128, 64, 64])
        # vector = vector.reshape(vector.shape[0], vector.shape[1], vector.shape[2], vector.shape[3])
        vector = self.multi_att1(
            vector.reshape(vector.shape[0], vector.shape[1], vector.shape[2]))  # torch.Size([128, 64, 64])
        vector = torch.sum(vector, dim=2)  # torch.Size([128, 64])
        vector = self.bn(vector)

        return vector


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lpe = LPE()
        self.att = Attentionadj()
        self.multi_att2 = MultiHeadSelfAttention(160, 160, 160)
        self.cnn = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 5))
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.mlp1 = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 4)
        )
        self.adaverage_pool = nn.AdaptiveAvgPool2d(output_size=(64, 1))

    def forward(self, lvalues, lvector):
        # 池化操作降维
        # lvalues = self.adaverage_pool(lvalues)  # ([128, 64, 1])
        # lvalues = self.mlp1(lvalues.reshape(lvalues.shape[0], -1))

        # cnn block
        lvalues = self.cnn(lvalues.unsqueeze(2))
        lvalues = self.relu(self.bn(lvalues))
        lvalues = lvalues.squeeze(-1).squeeze(-1)

        lvector = lvector.permute(0, 1, 3, 4, 2)  # ([128, 64, 64, 2])
        lpls_pe = self.lpe(lvector)  # ([128, 64])

        output = self.att(lvalues.reshape(lvalues.shape[0], 64) + lpls_pe)
        output = self.mlp2(output)

        return output
