import torch
import torch.nn.functional as F

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)           #L2归一化 torch.Size([nodes, 7, 10, 10])
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha) #逐元素乘积，沿着最后一个维度求和（-1,1）->（0,2）相同0，正交1，反向2 ；幂运算

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss