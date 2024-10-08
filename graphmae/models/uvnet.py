import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import NNConv
from dgl.nn.pytorch.glob import MaxPooling


class UVNetGraph(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraph, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,         # 修改
                    # out_feats=node_feats,       
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )
            self.edge_conv_layers.append(
                _EdgeConv(
                    edge_feats=edge_feats,
                    out_feats=hidden_dim,          # 修改
                    # out_feats=edge_feats,  
                    node_feats=hidden_dim,       # _NodeConv后的node_feats是out_feats，即hidden_dim
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                )
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = MaxPooling()

        self.linear_out1 = nn.Linear(hidden_dim, output_dim)
        self.linear_out2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, h, efeat):
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h, he)      # _NodeConv   torch.Size([nodes, 64])
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he)     # _EdgeConv 
            hidden_rep.append(h)                        # 输出的h和he特征数都为64

        # out = hidden_rep[-1]
        # out = self.drop1(out)
        # score_over_layer = 0

        # # Perform pooling over all nodes in each graph in every layer
        # for i, h in enumerate(hidden_rep):
        #     pooled_h = self.pool(g, h)
        #     score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        # return out, score_over_layer                #torch.Size([nodes, 64])      torch.Size([(batch_size, 256])

        V_emb = self.linear_out1(hidden_rep[-1])
        E_emb = self.linear_out2(he)

        return V_emb, E_emb
    
class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x                                              # torch.Size([nodes, 64]) 修改后：torch.Size([nodes, 256])
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h))) #Linear(in_features=256, out_features=64, bias=True) Linear(in_features=256, out_features=64, bias=True)
            return self.linears[-1](h)

class _EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.

        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_EdgeConv, self).__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):         # torch.Size([nodes, 64]), torch.Size([edges, 256])
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst]) # torch.Size([edges, 256])/64
        agg = proj1 + proj2                         
        h = self.mlp((1 + self.eps) * efeat + agg)  # torch.Size([edges, 64])
        h = F.leaky_relu(self.batchnorm(h))
        return h


class _NodeConv(nn.Module):
    def __init__(
        self,
        node_feats,           
        out_feats,           # 修改成了node_feats
        edge_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 1 from the paper where the node features are
        updated using the neighboring node and edge features.

        Args:
            node_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension  
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_NodeConv, self).__init__()
        self.gconv = NNConv(
            in_feats=node_feats,
            out_feats=out_feats,                                     # 64
            edge_func=nn.Linear(edge_feats, node_feats * out_feats), # Linear(in_features=256, out_features=256*64, bias=True)
            # edge_func=nn.Linear(edge_feats, node_feats * node_feats), 
            aggregator_type="sum",
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        # self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.mlp = _MLP(num_mlp_layers, hidden_mlp_dim, hidden_mlp_dim, out_feats) #修改
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):  # torch.Size([nodes, 256])  torch.Size([edges, 256])
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)      # node_feats -> out_feats    torch.Size([nodes, 256])
        h = self.mlp(h)                      # node_feats -> hidden_mlp_dim -> out_feats
        h = F.leaky_relu(self.batchnorm(h))
        return h

