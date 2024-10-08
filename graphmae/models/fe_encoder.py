import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair

from graphmae.utils import create_activation

class FaceEncoder(nn.Module):                                              # 传入V/face的grid和geom原始特征，输出face_emb
    def __init__(
            self, 
            # input_features = ["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=256,                                    # 嵌入维度大小，默认值是 384
            bias=False,
            dropout=0.0,
            grid_feature_size = 7,
            geom_feature_size = 7,
        ):
        super().__init__()

        # Calculate the total size of each feature list     # 根据 input_features 列表来确定哪些特征会被使用，并将它们映射到 self.grid_features 和 self.geom_features 中。
        self.grid_feature_size = grid_feature_size        # grid_features网格属性相关的特征索引大小        [0,1,2,3,4,5,6]
        self.geom_feature_size = geom_feature_size          # geom_features_size几何特征的总维度大小，不同属性维度大小不一样 7
        # self.grid_feat_v = grid_feat_v
        # self.geom_feat_v = geom_feat_v

        # Setup the layers                                  # 设置网络层：self.grid_embd 和 self.geom_embd
        self.emb_dim = emb_dim                              # 384
        if self.grid_feature_size > 0:                     # 如果有网格特征，则设置一个 CNN 层 self.grid_embd    7,384,384,3
            self.grid_embd = cnn2d(self.grid_feature_size, emb_dim, emb_dim, num_layers=3) 
        if self.geom_feature_size > 0:                      # 如果有几何特征，则设置一个线性层 self.geom_embd     7,384
            self.geom_embd = nn.Linear(self.geom_feature_size, emb_dim)
        self.ln = LayerNorm(emb_dim, bias=bias)             # 层归一化层 self.ln
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)     # 多层感知器 self.mlp (384,384)

    def forward(self, bg):
        x = torch.zeros(bg.num_nodes(), self.emb_dim, dtype=torch.float, device=bg.ndata['grid'].device)    # 初始化一个零张量 x，其形状为 (节点数量, emb_dim)
        grid_feat_v = bg.ndata["grid"]                      # torch.Size([nodes, 7, 10, 10])
        geom_feat_v = bg.ndata["geom"]                      # torch.Size([nodes, 7])
        if self.grid_feature_size > 0:                                                                   # 存在网格特征，从输入数据中提取这些特征并传递给 self.grid_embd
            # grid_feat = bg.ndata['uv'][:, :, :, self.grid_features].permute(0, 3, 1, 2)
            x += self.grid_embd(grid_feat_v)                                                                # (Nv,256)
        if self.geom_feature_size > 0:                                                                    # 存在几何特征,传递给 self.geom_embd
            x += self.geom_embd(geom_feat_v)
        x = x + self.mlp(self.ln(x))                                                                       # 将所有得到的特征向量相加，并通过多层感知器 self.mlp 和层归一化 self.ln
        return x   # 返回最终的编码向量 x


class EdgeEncoder(nn.Module):                              # 传入E/edge的grid和geom原始特征，输出edge_emb

    def __init__(
            self, 
            # input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=256,
            bias=False,
            dropout=0.0,
            grid_feature_size = 6,
            geom_feature_size = 5,
        ):
        super().__init__()
        # Calculate the total size of each feature list
        self.grid_feature_size = grid_feature_size             # grid_features网格属性相关的特征索引大小        [0,1,2,3,4,5]
        self.geom_feature_size = geom_feature_size          # geom_features_size几何特征的总维度大小，不同属性维度大小不一样 
        # Setup the layers
        self.emb_dim = emb_dim
        if self.grid_feature_size > 0:
            self.grid_embd = cnn1d(self.grid_feature_size, emb_dim, emb_dim, num_layers=3)
        if self.geom_feature_size > 0:
            self.geom_embd = nn.Linear(self.geom_feature_size, emb_dim)
        self.ln = LayerNorm(emb_dim, bias)
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, bg):
        x = torch.zeros(bg.num_edges(), self.emb_dim, dtype=torch.float, device=bg.edata['grid'].device)
        grid_feat_e = bg.edata["grid"]
        geom_feat_e = bg.edata["geom"]
        if self.grid_feature_size > 0:  
            # grid_feat = bg.edata['uv'][:, :, self.grid_features].permute(0, 2, 1)
            x += self.grid_embd(grid_feat_e) 
        if self.geom_feature_size > 0:
            x += self.geom_embd(geom_feat_e)
        x = x + self.mlp(self.ln(x))
        return x                                     #(edges,256)
    



def cnn2d(inp_channels, hidden_channels, out_dim, num_layers=1):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv2d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


def cnn1d(inp_channels, hidden_channels, out_dim, num_layers=1):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv1d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool1d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(input_dim, 4 * input_dim, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)   #torch.Size([12336, 256])
        return x


class LayerNorm(nn.Module):

    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, n_dim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))                    # 指定归一化操作的维度大小（通常是特征维度）,初始值为全1张量
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None   # 如果为 True，则包含偏置项；如果为 False，则不包含偏置项;

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5) # 指定归一化的维度,(n_dim,); 缩放因子 ; 偏置项;数值稳定的小常数，防止除以0