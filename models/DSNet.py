import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_ops import pointnet2_utils
from utils.mm3d_pn2 import furthest_point_sample, gather_points
from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling
# from paconv_util.PAConv_util import get_scorenet_input, knn
from DGCNN_PAConv import PAConv
from .Voxelbranch import Voxel
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL1
from .Pointbranch import Point


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return sub_pc


# class RandomPointSampling(torch.nn.Module):
#     def __init__(self, n_points):
#         super(RandomPointSampling, self).__init__()
#         self.n_points = n_points
#
#     def forward(self, pred_cloud, partial_cloud=None):
#         if partial_cloud is not None:
#             pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)
#
#         _ptcloud = torch.split(pred_cloud, 1, dim=0)
#         ptclouds = []
#         for p in _ptcloud:
#             non_zeros = torch.sum(p, dim=2).ne(0)
#             p = p[non_zeros].unsqueeze(dim=0)
#             n_pts = p.size(1)
#             if n_pts < self.n_points:
#                 rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points,))])
#             else:
#                 rnd_idx = torch.randperm(p.size(1))[:self.n_points]
#             ptclouds.append(p[:, rnd_idx, :])
#
#         return torch.cat(ptclouds, dim=0).contiguous()


##################################################
# Positional encoding (section 5.1),
# From NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 2048,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


embed_local, input_ch_local = get_embedder(2)  # L=10


# print('input_ch_local is ',input_ch_local)
##################################################
class local_encoder(torch.nn.Module):
    def __init__(self):
        super(local_encoder, self).__init__()
        self.conv = PAConv()

    def forward(self, x):
        xyz = x
        conv = self.conv(xyz)
        local_feature = conv
        return local_feature  # b,2048


class local_decoder(torch.nn.Module):
    def __init__(self):
        super(local_decoder, self).__init__()
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * 2048)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x_local_feature):
        x_local_feature = F.relu(self.bn1(self.linear1(x_local_feature)))
        x_local_feature = F.relu(self.bn2(self.linear2(x_local_feature)))
        x_local_feature = self.linear3(x_local_feature)
        y_coarse = x_local_feature.view(-1, 3, 2048).transpose(1, 2)
        return y_coarse  # b,2048,3


class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

        self.trans_conv = nn.Conv1d(d_model_out, d_model_out, 1)
        self.after_norm = nn.BatchNorm1d(d_model_out)
        self.act = nn.ReLU()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        # x = src1
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        # src1 = F.normalize(src1, p=2, dim=-1)
        # src2 = F.normalize(src2, p=2, dim=-1)

        src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]

        # src12 = x - src12
        # src12 = self.act(self.after_norm(self.trans_conv(src12.permute(1, 2, 0))))  # 12 64 512
        # src1 = x + src12.permute(2, 0, 1)

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.relu = nn.GELU()
        self.sa0_d = cross_transformer(channel*8, channel*8)
        self.sa1_d = cross_transformer(channel*8, channel*8)
        self.sa2_d = cross_transformer(channel*8, channel*8)
        self.sa3_d = cross_transformer(channel*8, channel*8)

        self.conv_out = nn.Conv1d(64, 3, 1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, 1)

        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, 1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, 1)


    def forward(self, feature_global, xyz):
        # seed generator
        # maxpooling
        batch_size, N, _ = xyz.size()
        x = self.relu(self.ps_adj(feature_global))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa3_d(x1_d, x1_d)).reshape(batch_size, self.channel*4, N//8)

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        return feature_global, fine


class PCT_refine(nn.Module):
    def __init__(self, channel=128, rate=1):
        super(PCT_refine, self).__init__()
        self.ratio = rate
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel*2, 512)
        self.sa2 = cross_transformer(512, 512)
        self.sa3 = cross_transformer(512, channel*rate)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*rate, channel*rate, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, coarse, feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N*self.ratio)

        y_up = y.repeat(1, 1, self.ratio)
        y_cat = torch.cat([y3, y_up], dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)

        return x, y3


@MODELS.register_module()
class DSNet(torch.nn.Module):
    def __init__(self, config):
        super(DSNet, self).__init__()
        self.number_fine = config.num_pred
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_query = config.num_query
        self.point_encoder = Point(in_chans=3, embed_dim=self.trans_dim, depth=[6, 8], drop_rate=0.,
                                           num_query=self.num_query, knn_layer=self.knn_layer)
        self.grnet_ = Voxel()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1024, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(2048, 2048, 1),
        )

        self.decrease_dim = nn.Sequential(
            nn.Conv1d(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 512, 1)
        )

        if self.number_fine == 16384:
            first_module = 4
            second_module = 8
        elif self.number_fine == 8192: 
            first_module = 4
            second_module = 4
        elif self.number_fine == 2048: 
            first_module = 1
            second_module = 4
        else:
            first_module = 4
            second_module = 8
        print(first_module)
        print(second_module)

        self.encoder = PCT_encoder()
        self.refine = PCT_refine(rate=first_module)
        self.refine1 = PCT_refine(rate=second_module)

        self.conv66 = nn.Conv1d(1024, 512, 1)

        self.linear1 = nn.Linear(10240, 2048)
        self.linear2 = nn.Linear(10240, 2048)

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):  # b 2048 3
        # partial_cloud = data['partial_cloud'].contiguous()
        partial_cloud = xyz.contiguous()
        # print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        # exit()
        # xyz = partial_cloud.transpose(1, 2).contiguous()
        # idx, _ = knn(xyz, k=8)  # get the idx of knn in 3D space : b,n,k
        # xyz_score = get_scorenet_input(xyz, k=8, idx=idx)
        # conv = self.conv(xyz)  # b,2048,N
        
        # conv = self.local_encoder(xyz)  # b,2048
        conv, coarse_cloud = self.point_encoder(xyz)   # b,1024,1   b,2048,3
        conv = self.conv_1(conv)  # b,2048,1
        # f_p = conv.squeeze(2)
        f_p = embed_local(conv.squeeze(2))  # b,2048>10240
        f_p = self.linear2(f_p)  # b,2048
        # sparse_cloud = self.local_decoder(conv)  # b,2048,3
        
        # sparse_cloud, dense_cloud, f_v = self.grnet_(data)
        # s_cloud, d_cloud, f_v = self.grnet_(partial_cloud)
        f_v = self.grnet_(partial_cloud)  # b,2048
        
        f_v = embed_local(f_v)  # b,2048>10240
        f_v = self.linear1(f_v)  # b,2048
        # f = torch.cat([f_p, f_v], dim=1)  # b,4096
        f_pv = f_p + f_v  # b,2048
        f = f_pv.unsqueeze(2)  # b, 2048, 1
        feat_g = self.decrease_dim(f)  # b, 512, 1

        new_x = torch.cat([xyz.transpose(1, 2), coarse_cloud.transpose(1, 2)], dim=2)
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        # coarse_cd = torch.cat([sparse_cloud, partial_cloud], dim=1)
        # new_x = fps(coarse_cd, 512)
        # new_x = new_x.transpose(1, 2)

        fine, feat_fine = self.refine(None, new_x, feat_g)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

        coarse_point_cloud = new_x.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()

        coarse_point_clouds = coarse_point_cloud.contiguous()
        rebuild_points = fine1.contiguous()

        ret = (coarse_point_clouds, rebuild_points)

        return ret, f_p, f_v, f_pv

