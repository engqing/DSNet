import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from .dgcnn_group import Pct
from utils.logger import *
import numpy as np
from knn_cuda import KNN
import torch.nn.functional as F
knn = KNN(k=8, transpose_mode=False)


def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        _, idx = knn(coor_k, coor_q)  # bs k np
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)

    return idx  # bs*k*np


def get_graph_feature(x, knn_index, x_q=None):
    # x: bs, np, c, knn_index: bs*k*np
    k = 8
    batch_size, num_points, num_dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
    feature = feature.view(batch_size, k, num_query, num_dims)
    x = x_q if x_q is not None else x
    x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - x, x), dim=-1)
    return feature  # b k np c


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.trans_conv = nn.Conv1d(128, 128, 1)
        self.after_norm = nn.BatchNorm1d(128)
        self.act = nn.ReLU()

    def forward(self, x):
        B, N, C = x.shape
        # x1 = x
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q = F.normalize(q, p=2, dim=-2)
        # k = F.normalize(k, p=2, dim=-2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attention = (q @ k.transpose(-2, -1))
        # attention = attention.softmax(dim=-1)
        # attention = attention / (1e-9 + attention.sum(dim=-2, keepdim=True))
        #
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.act(self.after_norm(self.trans_conv(x1 - x)))
        # x = x1 + x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim * 2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index=None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)

        if knn_index is not None:
            knn_f = get_graph_feature(norm_x, knn_index)
            knn_f = self.knn_map(knn_f) 
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            x_1 = torch.cat([x_1, knn_f], dim=-1)
            x_1 = self.merge_map(x_1) 

        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x


class Point(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 6], num_heads=6, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 num_query=224, knn_layer=-1):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.knn_layer = knn_layer

        print_log(' Transformer with knn_layer %d' % self.knn_layer, logger='MODEL')

        self.grouper = Pct()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )
        # self.pos_embed_wave = nn.Sequential(
        #     nn.Conv1d(60, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(128, embed_dim, 1)
        # )

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])

        # self.increase_dim = nn.Sequential(
        #     nn.Linear(embed_dim,1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024)
        # )

        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.num_query = num_query
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query)
        )
        self.mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * 512)
        )

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64  #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda()
        freqs = np.pi * (2 ** freqs)

        freqs = freqs.view(*[1] * len(normal_coor.shape), -1)  # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1)  # B x 3 x N x 1
        k = normal_coor * freqs  # B x 3 x N x D
        s = torch.sin(k)  # B x 3 x N x D
        c = torch.cos(k)  # B x 3 x N x D
        x = torch.cat([s, c], -1)  # B x 3 x N x 2D
        pos = x.transpose(-1, -2).reshape(coor.shape[0], -1, coor.shape[-1])  # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

    def forward(self, inpc):

        bs = inpc.size(0)
        coor, f, local_feature = self.grouper(inpc.transpose(1, 2).contiguous())
        coor = coor.contiguous()
        knn_index = get_knn_index(coor)

        # pos = self.pos_encoding_sin_wave(coor).transpose(1,2)
        pos = self.pos_embed(coor).transpose(1, 2)
        x = self.input_proj(f).transpose(1, 2)
        # cls_pos = self.cls_pos.expand(bs, -1, -1)
        # cls_token = self.cls_pos.expand(bs, -1, -1)
        # x = torch.cat([cls_token, x], dim=1)
        # pos = torch.cat([cls_pos, pos], dim=1)

        for i, blk in enumerate(self.encoder): 
            if i < self.knn_layer:  
                x = blk(x + pos, knn_index)  # B N C
            else:
                x = blk(x + pos)

        # global_feature  = x[:, 0] # B C

        global_feature = self.increase_dim(x.transpose(1, 2))  # B 1024 N
        # global_feature = x.transpose(1, 2)
        global_feature = torch.max(global_feature, dim=-1, keepdim=True)[0]  # B 1024 1
        coarse_point_cloud = self.mlp(global_feature.squeeze(2)).reshape(-1, 512, 3)

        return global_feature, coarse_point_cloud


