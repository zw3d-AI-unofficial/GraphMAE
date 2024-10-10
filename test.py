import torch
import torch.nn as nn
from graphmae.datasets.data_util import load_graph_dataset
import dgl


def center_and_scale_graph(graph):
    graph.ndata['grid'] = g.ndata['grid'].permute(0, 2, 3, 1)         #torch.Size([nodes,7,10,10])->[nodes,10,10,7]
    graph.edata['grid'] = g.edata['grid'].permute(0, 2, 1)            #torch.Size([nodes,6,10])->[nodes,10,6]

    graph.ndata["grid"], center, scale = center_and_scale_uvgrid(     #torch.Size([nodes,10,10,7])
        graph.ndata["grid"], return_center_scale=True
    )
    graph.edata["grid"][..., :3] -= center                            # 对edata中的坐标进行相同的中心化和缩放
    graph.edata["grid"][..., :3] *= scale

    graph.ndata['grid'] = g.ndata['grid'].permute(0, 3, 1, 2)         #torch.Size([nodes,10,10,7])->[nodes,7,10,10]
    graph.edata['grid'] = g.edata['grid'].permute(0, 2, 1)            #torch.Size([nodes,10,6])->[nodes,6,10]
    return graph

def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):  #torch.Size([nodes, 10,10,7])
    bbox = bounding_box_uvgrid(inp)                    # box: torch.Size([2, 3])
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])       # 最大的边归一化到长度为2
    center = 0.5 * (bbox[0] + bbox[1])                 # box的中心
    inp[..., :3] -= center                             # 坐标点中心化和缩放
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp

def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))       # torch.Size([nodes, 10, 10, 7]) -> ([nodes, 10, 10, 3]) -> torch.Size([nodes*10*10, 3]) 取0,1,2列
    mask = inp[..., 6].reshape(-1)            # torch.Size([nodes*10*10])
    point_indices_inside_faces = mask == 1    # torch.Size([nodes*10*10])
    pts = pts[point_indices_inside_faces, :]  # mask为0的nodes就不要了 
    return bounding_box_pointcloud(pts)


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


dataset,_ = load_graph_dataset()

g = dataset[0][0]

# # 构建一个简单的图
# src_nodes = [0, 1, 2, 3, 2, 7, 8, 5, 9, 4] 
# dst_nodes = [1, 2, 3, 0, 7, 8, 4, 3, 9, 5]
# g = dgl.graph((src_nodes, dst_nodes), idtype=torch.int32)

# # 示例数据
# points = torch.rand((10, 3))  # 10个三维点                    torch.Size([nodes, 3])
# mask = torch.randint(0, 2, (10, 1))  # 每个点是否有效的掩码    torch.Size([nodes, 1]) [0,2)之间的整数
# uvgrid = torch.cat([points, torch.zeros(10, 3), mask], dim=1)  # 模拟 uvgrid 数据  3,3,1 torch.Size([nodes, 7])
# g.ndata['x'] = uvgrid

# # 给边添加坐标数据，这里我们简单地复制节点坐标作为边的属性
# edge_coords = torch.cat([uvgrid[src_nodes], uvgrid[dst_nodes]], dim=1)
# g.edata['x'] = edge_coords

graph = center_and_scale_graph(g)