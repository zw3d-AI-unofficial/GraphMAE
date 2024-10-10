
from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler

import os
from dgl.data.utils import load_graphs

GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()                         # 在内部创建图的各种存储格式，优化图的内存表示和提高图操作的性能
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def load_graph_dataset(dataset_name, deg4feat=False): #deg4feat：是否使用节点的度数作为特征
    folder_path = "/home/share/brep/fusion/joint/graph"
    dataset = []
    i = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.bin'):
            file_path = os.path.join(folder_path, file_name)
            graph = load_graphs(str(file_path))      #load_graphs(str(file_path))->(Graph,{}); <class 'list'>
            dataset.append(graph)   

            i+=1
        if i > 3000:
            break             

    new_dataset = []
    for j, (graph, label) in enumerate(dataset):
        graph = graph[0]
        # graph = dgl.to_bidirected(graph)                            # 将一个有向图转换为双向图
        graph = graph.remove_self_loop().add_self_loop()              # 添加自环
        label = torch.tensor(0, dtype=torch.int).unsqueeze(0)         # 为了后续分类任务添加标签0
        if graph.ndata['uv'].ndim != 4 or graph.edata['uv'].ndim != 3:
            continue
        graph = renew_face_feature(graph)
        graph = renew_edge_feature(graph)
        # 将修改后的图和标签添加到新的列表中
        new_dataset.append((graph, label))
    dataset = list(map(tuple, new_dataset))
    
    graph = dataset[0]
    grid_feat_v_dim = graph[0].ndata["grid"].size()
    geom_feat_v_dim = graph[0].ndata["geom"].size()
    grid_feat_e_dim = graph[0].edata["grid"].size()
    geom_feat_e_dim = graph[0].edata["geom"].size()


    print(f"******** # Num Graphs: {len(dataset)}, # V Num Feat: {grid_feat_v_dim}, {geom_feat_v_dim} ********")
    return dataset, ((grid_feat_v_dim[1:], geom_feat_v_dim[1:]),(grid_feat_e_dim[1:],geom_feat_e_dim[1:]))    # 元组存放4个feat的torch.Size([])



class JointGraphDataset():
        SURFACE_GEOM_FEAT_MAP = {
        "type": 6,
        # "parameter": 2,
        # "axis": 6,
        # "box": 6,
        "area": 1,
        # "circumference": 1
    }
        CURVE_GEOM_FEAT_MAP = {
            "type": 4,
            # "parameter": 2,
            # "axis": 6,
            # "box": 6,
            "length": 1
        }

def renew_face_feature(g):
    input_features=["type","area","length","points","normals","tangents","trimming_mask"]
    grid_features = []                                 # grid_features网格属性相关的特征索引           [0,1,2,3,4,5,6]
    if "points" in input_features:
        grid_features.extend([0, 1, 2])            
    if "normals" in input_features:
        grid_features.extend([3, 4, 5])
    if "trimming_mask" in input_features:
        grid_features.append(6)
   
    if len(grid_features) > 0:   
        grid_feat_v = g.ndata['uv'][:, :, :, grid_features].permute(0, 3, 1, 2) #返回的特征 torch.Size([14, 10, 10, 7]) -> torch.Size([14, 7, 10, 10])
    grid_feat_v = grid_feat_v.float()

    geom_features = []                             # geom_features几何属性相关的特征名称。          ["type"、"area"]
    geom_feature_size = 0                          # geom_features_size几何特征的总维度大小，不同属性维度大小不一样 7
    for feat, feat_size in JointGraphDataset.SURFACE_GEOM_FEAT_MAP.items():
        if feat in list(g.ndata.keys()):
            geom_features.append(feat)
            geom_feature_size += feat_size 

    if geom_feature_size > 0:                                                                    # 存在几何特征
        geom_feat_v_l = []
        for feat in geom_features:
            if feat == "type":
                feat = F.one_hot(g.ndata[feat], num_classes=JointGraphDataset.SURFACE_GEOM_FEAT_MAP["type"]) # 从输入数据中提取这些特征并进行适当的预处理（如 one-hot 编码）
            else:
                feat = g.ndata[feat]
                if len(feat.shape) == 1:
                    feat = feat.unsqueeze(1)
            geom_feat_v_l.append(feat)                                        #返回的特征列表torch.Size([14, 6])/torch.Size([14, 1])->torch.Size([14, 7])
    geom_feat_v = torch.cat(geom_feat_v_l, dim=1).float()
    graph = g.clone()  
    graph.ndata.clear()
    # 将 grid_features 和 geom_features 设置为 graph 的 ndata 属性
    graph.ndata['grid'] = grid_feat_v
    graph.ndata['geom'] = geom_feat_v
    # 检查 ndata 是否只有 'grid'、'geom' 属性
    assert set(graph.ndata.keys()) == {'grid','geom'}, "图的 ndata 属性不符合预期"

    return graph

def renew_edge_feature(g):
    input_features=["type","area","length","points","normals","tangents","trimming_mask"]
    # Calculate the total size of each feature list
    grid_features = []                                 # grid_features网格属性相关的特征索引           [0,1,2,3,4,5]
    if "points" in input_features:
        grid_features.extend([0, 1, 2])
    if "tangents" in input_features:
        grid_features.extend([3, 4, 5])
    
    if len(grid_features) > 0:  
        grid_feat_e = g.edata['uv'][:, :, grid_features].permute(0, 2, 1)       #返回的特征
    grid_feat_e = grid_feat_e.float()

    geom_features = []                                 # geom_features几何属性相关的特征名称。          ["type"、"length"]
    geom_feature_size = 0                              # geom_features_size几何特征的总维度大小，不同属性维度大小不一样 5
    for feat, feat_size in JointGraphDataset.CURVE_GEOM_FEAT_MAP.items():
        if feat in list(g.edata.keys()):
            geom_features.append(feat)
            geom_feature_size += feat_size  

    if  geom_feature_size > 0:
        geom_feat_e_l = []
        for feat in geom_features:
            if feat == "type":
                feat = F.one_hot(g.edata[feat], num_classes=JointGraphDataset.CURVE_GEOM_FEAT_MAP["type"])
            else:
                feat = g.edata[feat]
                if len(feat.shape) == 1:
                    feat = feat.unsqueeze(1)
            geom_feat_e_l.append(feat)                                        #返回的特征
    geom_feat_e = torch.cat(geom_feat_e_l, dim=1).float()
    graph = g.clone()  
    graph.edata.clear()
    # 将 grid_features 和 geom_features 设置为 g 的 ndata 属性
    graph.edata['grid'] = grid_feat_e
    graph.edata['geom'] = geom_feat_e
    # 检查 ndata 是否只有 'grid'、'geom' 属性
    assert set(graph.edata.keys()) == {'grid','geom'}, "图的 ndata 属性不符合预期"

    return graph

