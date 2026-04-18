"""
二分类损失函数公式:
    loss = -ylog(y(预测)) - (1-y)log(1-y(预测))
"""

#导入库函数
import torch
import torch.nn as nn

#定义真实值(y)
y_true = torch.tensor([1, 0, 0],dtype = torch.float)
#定义预测值y_pred
y_pred = torch.rand(size=(1, 3), dtype = torch.float)
print(y_pred.data)
#定义二分类损失函数    return->平均值(mean)
creterion = nn.BCELoss()
loss = creterion(y_pred.data[0], y_true)
print(f'最终损失值为:{loss}')