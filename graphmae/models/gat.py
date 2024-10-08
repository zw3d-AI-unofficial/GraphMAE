import torch
import torch.nn as nn

from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair

from graphmae.utils import create_activation


class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,     # GAT 层的数量
                 nhead,          # 隐藏层中的注意力头数
                 nhead_out,      # 输出层中的注意力头数
                 activation,     # 激活函数
                 feat_drop,      # 特征的 dropout 概率
                 attn_drop,      # 注意力权重的 dropout 概率
                 negative_slope, # LeakyReLU 的负斜率
                 residual,       # 是否使用残差连接
                 norm,
                 concat_out=False,#是否拼接多头输出
                 encoding=False  # 是否用于编码任务
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None
        
        if num_layers == 1:                              # 根据层数的不同，初始化不同的 GATConv 层；
            self.gat_layers.append(GATConv(
                in_dim, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(              # 第一层不使用残差连接
                in_dim, num_hidden, nhead,
                feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):           # 中间层使用相同的配置
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * nhead, num_hidden, nhead,
                    feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # output projection
            self.gat_layers.append(GATConv(              # 最后一层根据是否用于编码任务来设置激活函数、残差连接和归一化层
                num_hidden * nhead, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))

        # if norm is not None:                           # 如果设置了归一化层，则初始化归一化层列表
        #     self.norms = nn.ModuleList([
        #         norm(num_hidden * nhead)
        #         for _ in range(num_layers - 1)
        #     ])
        #     if self.concat_out:
        #         self.norms.append(norm(num_hidden * nhead))
        # else:
        #     self.norms = None
    
        self.head = nn.Identity()                        # 初始化输出层为 Identity 层，表示默认情况下不改变输出

    # def forward(self, g, inputs):
    #     h = inputs
    #     for l in range(self.num_layers):
    #         h = self.gat_layers[l](g, h)
    #         if l != self.num_layers - 1:
    #             h = h.flatten(1)
    #             if self.norms is not None:
    #                 h = self.norms[l](h)
    #     # output projection
    #     if self.concat_out:
    #         out = h.flatten(1)
    #         if self.norms is not None:
    #             out = self.norms[-1](out)
    #     else:
    #         out = h.mean(1)
    #     return self.head(out)
    
    def forward(self, g, inputs, return_hidden=False): #输入为图 g 和节点特征 inputs；torch.Size([2708, 1433])
        h = inputs
        hidden_list = []                               
        for l in range(self.num_layers):               #通过每一层的 GATConv 层进行特征变换，存储每一层的输出到 hidden_list 中
            h = self.gat_layers[l](g, h)
            hidden_list.append(h)
            # h = h.flatten(1)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)



class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._concat_out = concat_out

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        # if norm is not None:
        #     self.norm = norm(num_heads * out_feats)
        # else:
        #     self.norm = None
    
        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')                      # 初始化权重时使用的增益因子，这里是针对ReLU激活函数计算的
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)      # 使用Xavier初始化方法初始化权重，这是一种均匀分布的初始化方法
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)            
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)                        # bias: 如果存在偏置项，则将其初始化为0。
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):                 # 设置是否允许图中有入度为0的节点
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):                     # 根据输入特征的类型（单个张量或元组），处理源节点和目标节点的特征
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])             # nn.Dropout
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):             # 使用线性变换（fc, fc_src, fc_dst）对节点特征进行变换，并将其重塑为多头形式
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]          #特征维度：torch.Size([2708]) torch.Size([2708, 1433])
                h_src = h_dst = self.feat_drop(feat)                           #存储了经过特征dropout处理后的源节点和目标节点的特征表示
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)       #重塑成多头的形式，即形状为 (num_nodes, num_heads, out_features)   torch.Size([2708, 4, 64])
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)             # 计算源节点和目标节点的注意力分数， torch.Size([2708, 4, 1]) (num_nodes, num_heads, 1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)             # torch.Size([2708, 4, 64])*torch.Size([1, 4, 64]),沿着维度（dim=-1）求和
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))                      # 在每条边上应用 fn.u_add_v 函数，将源节点的注意力分数 el 与目标节点的注意力分数 er 相加，得到每条边的注意力分数 e
            e = self.leaky_relu(graph.edata.pop('e'))                           # torch.Size([13264, 4, 1])
            # e[e == 0] = -1e3
            # e = graph.edata.pop('e')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))           # 使用 edge_softmax 计算边的注意力权重，并通过消息传递机制更新节点特征。
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),                        # 源节点的特征ft(num_nodes, num_heads, out_feats)与边的注意力权重 a(num_edges, num_heads) 相乘，得到消息 m
                             fn.sum('m', 'ft'))                                 # 将所有入边的消息 m(num_edges, num_heads, out_feats) 沿着边的维度进行求和，得到聚合后的特征表示 ft(num_nodes, num_heads, out_feats)
            rst = graph.dstdata['ft']

            # bias
            if self.bias is not None:                                                        # 如果存在偏置项，则添加偏置
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # residual
            if self.res_fc is not None:                                                      # 如果存在残差连接，则添加残差项
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            if self._concat_out:                                                             # 如果需要拼接多头输出，则展平输出；否则，计算多头的平均值
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:            # 如果存在规范化层，则应用规范化
                rst = self.norm(rst)

            # activation
            if self.activation:                  # 如果存在激活函数，则应用激活函数
                rst = self.activation(rst)

            if get_attention:                    # 是否返回注意力权重
                return rst, graph.edata['a']     # torch.Size([13264, 4, 1])
            else:
                return rst                       # torch.Size([2708, 256])
