from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT

from .fe_encoder import FaceEncoder, EdgeEncoder
from .uvnet_encoder import UVNetGraphEncoder

from .loss_func import sce_loss,loss_grid
from graphmae.utils import create_norm, drop_edge

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0),  # (Nv, 256, 1, 1) 输出大小: Nv x 128 x 4 x 4
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=1),   # 输出大小: Nv x 64 x 5 x 5
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=7, kernel_size=4, stride=2, padding=1),     # 输出大小: Nv x 7 x 10 x 10
            nn.Sigmoid()  # 输出范围在[0, 1] 7*10*10
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )
    def forward(self, x):          # (Nv,256)
        x1 = x.view(-1, 256, 1, 1) # 将 (Nv, 256) 转换为 (Nv, 256, 1, 1) 
        x1 = self.deconv(x1)       # 上采样
        x2 = self.mlp(x)           # (Nv,256) -> (Nv,7)
        return x1,x2
    
class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: tuple,              #((grid_feat_v_dim, geom_feat_v_dim),(grid_feat_e_dim,geom_feat_e_dim))
            num_hidden: int,
            mask_rate: float = 0.3,
            loss_fn: str = "test",         #修改/sce
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden #256
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate


        self.face_encoder = FaceEncoder(
            # input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=256,                                    # 嵌入维度大小，默认值是 384
            bias=False,
            dropout=0.0,
            grid_feature_size = in_dim[0][0][0],
            geom_feature_size = in_dim[0][1][0]
        )
        self.edge_encoder = EdgeEncoder(
            # input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=256,
            bias=False,
            dropout=0.0,
            grid_feature_size = in_dim[1][0][0],
            geom_feature_size = in_dim[1][1][0]
        )
        self.uvnet_encoder = UVNetGraphEncoder(
            input_dim = 256,
            input_edge_dim = 256,
            output_dim = 256,
            hidden_dim=64,
            learn_eps=True,
            num_layers=3,
            num_mlp_layers=2,            
        )
        self.decoder = Decoder()

        self.enc_mask_token_grid = nn.Parameter(torch.zeros(1, *in_dim[0][0]))     #grid_feat_v_dim
        self.enc_mask_token_geom = nn.Parameter(torch.zeros(1, *in_dim[0][1]))     #geom_feat_v_dim

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "test":
            criterion = partial(loss_grid) 
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)  # 根据sce_loss创建一个新的函数，这个函数在调用时已经预设了alpha的值
        else:
            raise NotImplementedError
        return criterion
    
    """对特征x进行编码掩码"""
    def encoding_mask_noise(self, x1,x2,mask_rate=0.3):   
        num_nodes = x1.shape[0]                     # 2708/    x.shape[0] nodes数
        perm = torch.randperm(num_nodes, device=x1.device) # perm 是一个随机排列的索引列表，用于后续选择掩码节点

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)       # 2708*0.3=1354
        mask_nodes = perm[: num_mask_nodes]               # mask的node索引
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:                        # 引入噪声；替换一部分掩码节点的特征
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)                          # 要替换的节点数量 
            perm_mask = torch.randperm(num_mask_nodes, device=x1.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]  # 要设置为掩码标记的节点索引
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x1.device)[:num_noise_nodes]   # 从所有节点中随机选取的用于替换的节点索引

            out_x1 = x1.clone()
            out_x2 = x2.clone()
            out_x1[token_nodes] = 0.0
            out_x1[noise_nodes] = x1[noise_to_be_chosen]
            out_x2[token_nodes] = 0.0
            out_x2[noise_nodes] = x2[noise_to_be_chosen]
        else:                                            # _replace_rate 为0，则所有掩码节点都将被设为0。
            
            out_x1 = x1.clone()
            out_x2 = x2.clone()
            token_nodes = mask_nodes
            out_x1[mask_nodes] = 0.0
            out_x2[mask_nodes] = 0.0

        out_x1[token_nodes] += self.enc_mask_token_grid     # self.enc_mask_token 是一个特定的掩码标记，它会被加到掩码节点的特征上。
        out_x2[token_nodes] += self.enc_mask_token_geom

        return (out_x1,out_x2),(mask_nodes, keep_nodes)    # 掩码后的node特征矩阵out_x；掩码节点和保留节点的索引

    def forward(self, g):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, g): #g: 输入的图对象。 x: 节点特征矩阵，形状为 (num_nodes, feature_dim)；返回重构误差 loss
        grid_feat_v = g.ndata["grid"]  # (Nv,7,10,10)
        geom_feat_v = g.ndata["geom"]  # (Nv,7)

        # 对node掩码/mask
        (xm1,xm2), (mask_nodes, keep_nodes) = self.encoding_mask_noise(grid_feat_v,geom_feat_v,self._mask_rate) # 掩码后的node特征矩阵；掩码节点和保留节点的索引
        pre_use_g = g.clone()                                                                   # 返回g的副本pre_use_g；
        pre_use_g.ndata.clear()
        pre_use_g.ndata['grid'] = xm1
        pre_use_g.ndata['geom'] = xm2

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)      #按照self._drop_edge_rate删除图中的mask的边，添加自环
        else:
            use_g = pre_use_g

        face_emb = self.face_encoder(use_g)                                     # (Nv,256)
        edge_emb = self.edge_encoder(use_g)                                     # (Ne,256)
        V_emb, E_emb = self.uvnet_encoder(use_g,face_emb,edge_emb)              # GCN,(Nv,256),(Ne,256)

        V_emb[mask_nodes] = 0                                                    #remask
        V_emb, E_emb = self.uvnet_encoder(use_g,V_emb,E_emb)                     # GCN Decoder,(Nv,256),(Ne,256)
        grid_rec,geom_rec = self.decoder(V_emb)                                        # 重构为原始特征

        x1_init = grid_feat_v[mask_nodes]                                                # 原始节点特征中被掩码的部分
        x2_init = geom_feat_v[mask_nodes]                                                # 原始节点特征中被掩码的部分

        x1_rec = grid_rec[mask_nodes]                                             # 重构后的节点特征中对应掩码部分的特征
        x2_rec = geom_rec[mask_nodes]                                             # 重构后的节点特征中对应掩码部分的特征

        loss1 = self.criterion(x1_rec, x1_init)
        loss2 = self.criterion(x2_rec, x2_init)
        return loss1+loss2                                                        # 返回重构误差 loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()]) #返回的是一个合并了 encoder_to_decoder 和 decoder 部分所有参数的迭代器
    

