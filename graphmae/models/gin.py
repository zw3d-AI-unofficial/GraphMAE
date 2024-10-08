import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair

from graphmae.utils import create_activation, NormLayer, create_norm


class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim         # 输出特征维度。
        self.num_layers = num_layers   # 网络的层数。
        self.layers = nn.ModuleList()  # 一个 ModuleList 用来存储每一层的 GINConv 实例
        self.activation = activation   # 激活函数
        self.dropout = dropout         # Dropout 概率

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:            # num_layers 等于 1 时，只创建一个 GINConv 层
            apply_func = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(GINConv(in_dim, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))
        else:                          # 当 num_layers 大于 1 时，创建多个 GINConv 层
            # input projection (no residual)
            self.layers.append(GINConv(               # 第一层没有残差连接
                in_dim, 
                num_hidden, 
                ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                init_eps=0,
                learn_eps=learn_eps,
                residual=residual)
                )
            # hidden layers
            for l in range(1, num_layers - 1):        # 中间层（除了第一层和最后一层外）的 apply_func 为一个 MLP 层
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GINConv(
                    num_hidden, num_hidden, 
                    ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                    init_eps=0,
                    learn_eps=learn_eps,
                    residual=residual)
                )
            # output projection                       # 最后一层根据 encoding 参数决定是否使用残差连接，并且可能应用额外的归一化。
            apply_func = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            self.layers.append(GINConv(num_hidden, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):  # g num_nodes=710, num_edges=12534, ndata:'ID'、'attr'
        h = inputs                                      # torch.Size([710, 271])
        hidden_list = []
        for l in range(self.num_layers):                # 对每一层的输入应用 F.dropout 并通过该层的 GINConv
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)                    # GINConv 实例,调用 GINConv 类的 forward 方法,赋值了apply_func的列表 torch.Size([710, 256])
            hidden_list.append(h)                       #  torch.Size([710, 271])
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes) # 更改 GIN 网络的输出层（分类头），使其能够适应不同数量的类别


class GINConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 apply_func,
                 aggregator_type="sum",
                 init_eps=0,
                 learn_eps=False,
                 residual=False,
                 ):
        super().__init__()
        self._in_feats = in_dim                            # 输入特征维度
        self._out_feats = out_dim                          # 输出特征维度
        self.apply_func = apply_func                       # 应用在节点特征上的函数，通常是一个神经网络层，如MLP

        self._aggregator_type = aggregator_type            # 聚合类型,默认为 "sum"，可以选择 "max" 或 "mean"
        if aggregator_type == 'sum':
            self._reducer = fn.sum                         # 根据聚合类型选择的 reducer 函数
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
            
        if learn_eps:                                      # 是否使 ε 成为可学习的参数，默认为 False,否则，它是一个固定的缓冲区。
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:                                              
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        if residual:                                       # 是否启用残差连接，默认为 False
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False) # 输入输出维度不匹配，则创建一个线性层用于调整维度；如果输入输出维度匹配
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        with graph.local_scope():                          # 确保在本次调用中的任何改变不会影响到图的原始状态。
            aggregate_fn = fn.copy_u('h', 'm')             # 修改：fn.copy_src -> fn.copy_u
                                                           # 消息函数,从源节点（源节点在 DGL 中通常标记为 'u'）复制其特征向量 h，并将这些特征作为消息 m 发送给目标节点
            feat_src, feat_dst = expand_as_pair(feat, graph)#扩展输入特征 feat 以便与图中的源节点和目标节点相匹配 torch.Size([710, 256])
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh')) #aggregate_fn 收集消息，然后使用 self._reducer 函数对收集到的消息进行聚合，最终结果保存在 graph.dstdata['neigh'] 中。
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh'] # 计算节点的新特征表示 rst，包括将节点自身的特征feat_dst与聚合后的邻居特征组合起来，并根据 apply_func 应用非线性变换。
            if self.apply_func is not None:                # ApplyNodeFunc(
                rst = self.apply_func(rst)               

            if self.res_fc is not None:                    # 如果有残差连接，则加上残差项
                rst = rst + self.res_fc(feat_dst)

            return rst


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = create_norm(norm)                       #根据传入的norm参数创建相应的归一化函数。如果norm无效，则使用nn.Identity()作为默认归一化层
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:                                               # num_layers是否为正数，如果不是则抛出异常
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:                                            # 如果num_layers为1，则创建一个线性模型
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()                     # 存储在ModuleList中

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):                         # 多层感知器模型，则遍历每一层，依次执行线性变换、归一化和激活操作
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)                                   # 返回最后一层线性变换的结果