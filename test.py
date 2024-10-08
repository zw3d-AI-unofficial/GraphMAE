import torch
import torch.nn as nn

# 假设 x 是模型的输出，y 是真实的图像
x = torch.randn(1, 3, 64, 64)
y = torch.randn(1, 3, 64, 64)

# 创建 MSELoss 实例
mse_loss = nn.MSELoss()

# 计算损失
loss = mse_loss(x, y)

# 输出损失值
print("MSE Loss:", loss)