import torch
import torch.nn as nn
import torch.nn.functional as F
# from datasets.fusion_joint import JointGraphDataset
from dgl.nn import GATv2Conv

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
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):

    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, n_dim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class FaceEncoder(nn.Module):

    def __init__(
            self, 
            input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=384,                                    # 嵌入维度大小，默认值是 384
            bias=False,
            dropout=0.0
        ):
        super().__init__()
        # Calculate the total size of each feature list     # 根据 input_features 列表来确定哪些特征会被使用，并将它们映射到 self.grid_features 和 self.geom_features 中。
        self.grid_features = []                             # grid_features网格属性相关的特征索引           [0,1,2,3,4,5,6]
        if "points" in input_features:
            self.grid_features.extend([0, 1, 2])            
        if "normals" in input_features:
            self.grid_features.extend([3, 4, 5])
        if "trimming_mask" in input_features:
            self.grid_features.append(6)
        self.geom_features = []                             # geom_features几何属性相关的特征名称。          ["type"、"area"]
        self.geom_feature_size = 0                          # geom_features_size几何特征的总维度大小，不同属性维度大小不一样 7
        for feat, feat_size in JointGraphDataset.SURFACE_GEOM_FEAT_MAP.items():
            if feat in input_features:
                self.geom_features.append(feat)
                self.geom_feature_size += feat_size

        # Setup the layers                                  # 设置网络层：self.grid_embd 和 self.geom_embd
        self.emb_dim = emb_dim                              # 384
        if len(self.grid_features) > 0:                     # 如果有网格特征，则设置一个 CNN 层 self.grid_embd    7,384,384,3
            self.grid_embd = cnn2d(len(self.grid_features), emb_dim, emb_dim, num_layers=3) 
        if self.geom_feature_size > 0:                      # 如果有几何特征，则设置一个线性层 self.geom_embd     7,384
            self.geom_embd = nn.Linear(self.geom_feature_size, emb_dim)
        self.ln = LayerNorm(emb_dim, bias=bias)             # 层归一化层 self.ln
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)     # 多层感知器 self.mlp (384,384)

    def forward(self, bg):
        x = torch.zeros(bg.num_nodes(), self.emb_dim, dtype=torch.float, device=bg.ndata['uv'].device)    # 初始化一个零张量 x，其形状为 (节点数量, emb_dim)
        if len(self.grid_features) > 0:                                                                   # 存在网格特征，从输入数据中提取这些特征并传递给 self.grid_embd
            grid_feat = bg.ndata['uv'][:, :, :, self.grid_features].permute(0, 3, 1, 2)
            x += self.grid_embd(grid_feat)                                                                # (Nv,256)
        if self.geom_feature_size > 0:                                                                    # 存在几何特征,传递给 self.geom_embd
            geom_feat = []
            for feat in self.geom_features:
                if feat == "type":
                    feat = F.one_hot(bg.ndata[feat], num_classes=JointGraphDataset.SURFACE_GEOM_FEAT_MAP["type"]) # 从输入数据中提取这些特征并进行适当的预处理（如 one-hot 编码）
                else:
                    feat = bg.ndata[feat]
                    if len(feat.shape) == 1:
                        feat = feat.unsqueeze(1)
                geom_feat.append(feat)
            x += self.geom_embd(torch.cat(geom_feat, dim=1).float())
        x = x + self.mlp(self.ln(x))                                                                       # 将所有得到的特征向量相加，并通过多层感知器 self.mlp 和层归一化 self.ln
        return x   # 返回最终的编码向量 x


class EdgeEncoder(nn.Module):

    def __init__(
            self, 
            input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=384,
            bias=False,
            dropout=0.0
        ):
        super().__init__()
        # Calculate the total size of each feature list
        self.grid_features = []
        if "points" in input_features:
            self.grid_features.extend([0, 1, 2])
        if "tangents" in input_features:
            self.grid_features.extend([3, 4, 5])
        self.geom_features = []
        self.geom_feature_size = 0
        for feat, feat_size in JointGraphDataset.CURVE_GEOM_FEAT_MAP.items():
            if feat in input_features:
                self.geom_features.append(feat)
                self.geom_feature_size += feat_size

        # Setup the layers
        self.emb_dim = emb_dim
        if len(self.grid_features) > 0:
            self.grid_embd = cnn1d(len(self.grid_features), emb_dim, emb_dim, num_layers=3)
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
        x = torch.zeros(bg.num_edges(), self.emb_dim, dtype=torch.float, device=bg.edata['uv'].device)
        if len(self.grid_features) > 0:  
            grid_feat = bg.edata['uv'][:, :, self.grid_features].permute(0, 2, 1)
            x += self.grid_embd(grid_feat) 
        if self.geom_feature_size > 0:
            geom_feat = []
            for feat in self.geom_features:
                if feat == "type":
                    feat = F.one_hot(bg.edata[feat], num_classes=JointGraphDataset.CURVE_GEOM_FEAT_MAP["type"])
                else:
                    feat = bg.edata[feat]
                    if len(feat.shape) == 1:
                        feat = feat.unsqueeze(1)
                geom_feat.append(feat)
            x += self.geom_embd(torch.cat(geom_feat, dim=1).float())
        x = x + self.mlp(self.ln(x))
        return x

# class JointGraphDataset(JointBaseDataset):
class JointGraphDataset():
        SURFACE_GEOM_FEAT_MAP = {
        "type": 6,
        "parameter": 2,
        "axis": 6,
        "box": 6,
        "area": 1,
        "circumference": 1
    }
        CURVE_GEOM_FEAT_MAP = {
            "type": 4,
            "parameter": 2,
            "axis": 6,
            "box": 6,
            "length": 1
        }

