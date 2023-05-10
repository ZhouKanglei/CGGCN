#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/5/2 下午7:01


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from graph.tools import k_adjacency, normalize_adjacency_matrix
from utils import import_class, model_params


def res_block(in_channels, out_channels, residual=True, stride=(1, 1), kernel_size=(1, 1)):
    """Residual block"""

    if not residual:
        res = lambda x: 0
    elif in_channels == out_channels:
        res = lambda x: x
    else:
        res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    return res


class mlp(nn.Module):
    "Multi-layer perceptron"

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1)):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.mlp(x)


class st_joint_att(nn.Module):
    """Spatial-temporal joint attention"""

    def __init__(self, channel, reduct_ratio=2, bias=True, **kwargs):
        super(st_joint_att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=(1, 1), bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=(1, 1))
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=(1, 1))

        self.data_bn = nn.BatchNorm2d(channel)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att

        out = x * x_att + x
        out = self.act(self.data_bn(out))

        return out


class st_att(nn.Module):
    """Spatial-temporal joint attention"""

    def __init__(self, channel, reduct_ratio=2, bias=True, groups=1):
        super(st_att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel * 2, inner_channel, kernel_size=(1, 1), bias=bias, groups=groups),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=(1, 1), groups=groups)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=(1, 1), groups=groups)

        self.data_bn = nn.BatchNorm2d(channel)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        N, C, T, V = x.size()
        x_at, x_mt = x.mean(3, keepdims=True), F.max_pool2d(x, (1, V))
        x_av, x_mv = x.mean(2, keepdims=True).transpose(2, 3), F.max_pool2d(x, (T, 1)).transpose(2, 3)
        h_t, h_v = self.fcn(torch.cat([x_at, x_mt], axis=1)), self.fcn(torch.cat([x_av, x_mv], axis=1))
        x_t_att = self.conv_t(h_t).sigmoid()
        x_v_att = self.conv_v((h_v).transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att

        out = x * x_att + x
        out = self.act(self.data_bn(out))

        return out

class gcn_att(nn.Module):
    """Graph neural network unit"""

    def __init__(self, in_channels, out_channels):
        super(gcn_att, self).__init__()

        embed_channels = out_channels // 2
        self.conv_q = nn.Conv2d(in_channels, embed_channels, kernel_size=(1, 1))
        self.conv_k = nn.Conv2d(in_channels, embed_channels, kernel_size=(1, 1))

        self.conv = nn.Conv2d(embed_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        q, k = self.conv_q(x).mean(2), self.conv_k(x).mean(2)
        C = F.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))
        C = self.conv(C)

        return C

class cross_att(nn.Module):
    """cross attention module unit"""

    def __init__(self, in_channels, num_scales):
        super(cross_att, self).__init__()

        self.num_scales = num_scales

        embed_channels = in_channels // 4
        self.conv_q = nn.Conv2d(in_channels * 2, embed_channels, kernel_size=(1, 1))
        self.conv_k = nn.Conv2d(in_channels, embed_channels, kernel_size=(1, 1))


    def forward(self, x1, x2):
        q, k = self.conv_q(x1).mean(2), self.conv_k(x2).mean(2)
        C = torch.einsum('n c v, n c u -> n u v', q, k)
        C = torch.softmax(C / np.sqrt(q.shape[1]), dim=-1)

        C = C.repeat(1, self.num_scales, 1)

        return C


class ms_gcn_agg(nn.Module):
    """Multi-scale group graph aggregation"""

    def __init__(self, in_channels, out_channels, A, num_scales, num_groups=3, stride=1,
                 adaptive=True, attention=False, residual=False, disentangled_agg=True):
        super().__init__()

        self.num_scales, self.num_groups = num_scales, num_groups
        self.attention = attention

        if disentangled_agg:
            A_powers = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_mask = np.concatenate(A_powers)
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A + np.eye(len(A)) for _ in range(num_scales)]
            A_mask = np.concatenate(A_powers)
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A, self.A_mask = torch.Tensor(A_powers), torch.Tensor(A_mask)

        # adaptive adj matrix
        self.B = nn.init.uniform_(nn.Parameter(torch.zeros(self.A.shape)), -1e-6, 1e-6) if \
            adaptive else torch.zeros(self.A.shape)

    def forward(self, x):
        n, c, t, v = x.shape

        if self.attention:
            A_sum = self.A.to(x.device) + self.B.to(x.device) * self.A_mask.to(x.device)
        else:
            A_sum = self.A.to(x.device) + self.B.to(x.device)

        h = torch.einsum('vu, nctu->nctv', A_sum, x)
        y = h.view(n, c, t, self.num_scales, v).mean(3)

        return y

class group_agg(nn.Module):

    def __init__(self, num_groups):
        super(group_agg, self).__init__()

        self.num_groups = num_groups

        self.pool = nn.AvgPool2d(kernel_size=(num_groups, 1), stride=(1, 1),
                                     padding=(num_groups // 2, 0))

    def forward(self, x):
        pre_x = torch.zeros_like(x)
        pre_x[:, :, 1:, :] = x[:, :, :-1, :]
        diff = x - pre_x

        y = self.pool(diff)

        return y

class ms_gcn_group_cat(nn.Module):
    """Multi-scale group graph neural network"""

    def __init__(self, in_channels, out_channels, A, num_scales, num_groups=3, stride=1,
                 adaptive=True, attention=False, residual=False, disentangled_agg=True, c_att=False):
        super().__init__()

        self.num_scales, self.num_groups = num_scales, num_groups
        self.attention = attention

        if disentangled_agg:
            A_powers = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_mask = np.concatenate(A_powers)
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A + np.eye(len(A)) for _ in range(num_scales)]
            A_mask = np.concatenate(A_powers)
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A, self.A_mask = torch.Tensor(A_powers), torch.Tensor(A_mask)

        # adaptive adj matrix
        self.B = nn.init.uniform_(nn.Parameter(torch.zeros(self.A.shape)), -1e-6, 1e-6) if \
            adaptive else torch.zeros(self.A.shape)

        # residual
        self.residual = res_block(in_channels, out_channels, residual=residual)

        # temporal aggregation
        self.group_agg = group_agg(num_groups)

        # cross attention
        self.cross_att = cross_att(in_channels, num_scales) if c_att is not False else c_att

        # feature learning
        self.conv = nn.Conv2d(in_channels * self.num_scales * 2 if num_groups >= 1 else
                              in_channels * self.num_scales, out_channels,
                              kernel_size=(1, 1), stride=(stride, 1), groups=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.residual(x)

        if self.attention:
            A_sum = self.A.to(x.device) + self.B.to(x.device) * self.A_mask.to(x.device)
        else:
            A_sum = self.A.to(x.device) + self.B.to(x.device)

        # group aggregation
        if self.num_groups > 0:
            x_trend = self.group_agg(x)
            h = torch.cat([x, x_trend], dim=1)
            c *= 2
            if self.cross_att is not False:
                A_sum = A_sum.unsqueeze(0) + self.B.to(x.device) * self.cross_att(h, x)

        else:
            h = x

        if self.cross_att is not False and self.num_groups > 0:
            h = torch.einsum('nvu, nctu->nctv', A_sum, h)
        else:
            h = torch.einsum('vu, nctu->nctv', A_sum, h)

        h = h.view(n, c, t, self.num_scales, v)
        h = h.permute(0, 3, 1, 2, 4).contiguous().view(n, self.num_scales * c, t, v)
        h = self.conv(h)

        y = self.act(self.bn(h) + res)
        return y


class tcn_unit(nn.Module):
    """Temporal convolution unit"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(tcn_unit, self).__init__()

        t_pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              dilation=(dilation, 1), stride=(stride, 1), padding=(t_pad, 0))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = self.conv(x)
        y = self.bn(h)

        return y

class mk_tcn(nn.Module):
    """Multi-kernel temporal convolution unit"""

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 3, 4), stride=1,
                 kernel_sizes=(3, 5, 7, 9), residual=True):
        super(mk_tcn, self).__init__()

        assert out_channels % (len(kernel_sizes) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(kernel_sizes) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=(1, 1),
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=False),
                tcn_unit(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=1
                )
            )
            for kernel_size in kernel_sizes
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1), stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        self.residual = res_block(in_channels, out_channels, residual=residual, stride=(stride, 1))

        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        res = self.residual(x)

        branch_outs = []
        for idx, tempconv in enumerate(self.branches):
            h = tempconv(x)
            branch_outs.append(h)

        out = torch.cat(branch_outs, dim=1)
        y = self.act(out + res)

        return y

class ms_tcn(nn.Module):
    """Multi-scale temporal convolution unit"""

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 3, 4), stride=1, kernel_size=3,
                 residual=True):
        super(ms_tcn, self).__init__()

        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=(1, 1),
                    padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=False),
                tcn_unit(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation
                )
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1), stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        self.residual = res_block(in_channels, out_channels, residual=residual, stride=(stride, 1))

        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        res = self.residual(x)

        branch_outs = []
        for idx, tempconv in enumerate(self.branches):
            h = tempconv(x)
            branch_outs.append(h)

        out = torch.cat(branch_outs, dim=1)
        y = self.act(out + res)

        return y

class st_gcn_unit(nn.Module):
    """Spatial-temporal graph convolution network block"""

    def __init__(self, in_channels, out_channels, A, num_gcn_scales=8, num_groups=1,
                 stride=1, attention=False, st_attention=True):
        super(st_gcn_unit, self).__init__()

        self.block_1 = nn.Sequential(
            ms_gcn_group_cat(in_channels, in_channels, A, num_gcn_scales, num_groups, attention=attention),
            ms_tcn(in_channels, out_channels, stride=stride),
            ms_tcn(out_channels, out_channels),
            st_att(out_channels, groups=2) if st_attention else nn.Identity()
        )

        self.block_2 = nn.Sequential(
            ms_gcn_group_cat(in_channels, in_channels, A, num_gcn_scales, num_groups, attention=attention),
            ms_tcn(in_channels, out_channels, stride=stride),
            ms_tcn(out_channels, out_channels),
            st_att(out_channels, groups=2) if st_attention else nn.Identity()
        )

        self.alpha, self.beta = nn.Parameter(torch.Tensor(1)), nn.Parameter(torch.Tensor(1))

    def forward(self, x_1, x_2):
        alpha, beta = self.alpha, self.beta
        x_1, x_2 = self.block_1(x_1), self.block_2(x_2)
        x_1_out = x_1 + alpha * x_2
        x_2_out = x_2 + beta * x_1

        return x_1_out, x_2_out


class Model(nn.Module):
    """Implementation of our model_msg3d"""

    def __init__(self, num_class, num_point, num_person, graph, in_channels=3,
                 num_gcn_scales=8, num_groups=(1, 1, 1), c1=96, groups=1):
        super(Model, self).__init__()

        self.num_class = num_class
        self.num_point = num_point

        Graph = import_class(graph)
        A = Graph().A_binary[0]

        self.data_bn_1 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_2 = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c2 = c1 * 2  # 192
        c3 = c2 * 2  # 384

        self.embed_1 = mlp(3, c1)
        self.embed_2 = mlp(3, c1)

        # 3 ST-GCN blocks
        self.st_gcn_1 = st_gcn_unit(c1, c1, A, num_gcn_scales, num_groups[0])
        self.st_gcn_2 = st_gcn_unit(c1, c2, A, num_gcn_scales, num_groups[1], stride=2)
        self.st_gcn_3 = st_gcn_unit(c2, c3, A, num_gcn_scales, num_groups[2], stride=2)

        self.fc_1 = nn.Linear(c3, num_class)
        self.fc_2 = nn.Linear(c3, num_class)

        self.gamma = nn.Parameter(torch.Tensor(1))

    def pre_process_1(self, h):
        x = h.clone()
        # normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn_1(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # embedding
        x = self.embed_1(x)

        return x

    def pre_process_2(self, h):
        x = h.clone()
        # normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn_2(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # embedding
        x = self.embed_2(x)

        return x

    def forward(self, h):
        # x is the modality 1; m: modality 2
        if h.dim() == 6:
            x, m = h[:, 0], h[:, 1]
            N, C, T, V, M = x.size()
            x, m = self.pre_process_1(x), self.pre_process_2(m)
        else:
            N, C, T, V, M = h.size()
            x = self.pre_process_1(h)
            frame = list(range(1, T))
            frame.append(T - 1)
            m = x[:, :, frame] - x
            m = (m[:, :, frame] + m) / 2

        # ST-GCN
        x, m = self.st_gcn_1(x, m)
        x, m = self.st_gcn_2(x, m)
        x_out, m_out = self.st_gcn_3(x, m)

        # output
        out_channels = x_out.size(1)
        x_out, m_out = x_out.view(N, M, out_channels, -1), m_out.view(N, M, out_channels, -1)
        x_out, m_out = x_out.mean(3), m_out.mean(3)  # Global Average Pooling (Spatial + Temporal)
        x_out, m_out = x_out.mean(1), m_out.mean(1)  # Average pool number of bodies in the sequence

        x_out, m_out = self.fc_1(x_out), self.fc_2(m_out)

        return x_out, m_out, x_out + m_out

if __name__ == '__main__':
    from torchsummary import summary

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=8,
        num_groups=(7, 7, 7),
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 64, 25, 2
    x = torch.randn(N, 2, C, T, V, M)
    model.forward(x)

    from torchsummary import summary

    summary(model, input_size=(2, 3, 300, 25, 2), device='cpu')

    from thop import profile

    flops, params = profile(model, inputs=(x.to('cpu'),))
    print(f'{flops / 1e9:.4f} GFLOPS')

    model_params(model)
    # print(model)
