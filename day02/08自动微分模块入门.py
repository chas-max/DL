"""
自动微分函数API:
backward()
公式：
w新=w旧-学习率*梯度
注：Pytorch提供了 自动求导函数backward()，并将梯度保存在 grid属性 中,自动求导函数仅支持对 标量张量 进行求导
"""

import torch

#定义w(旧)
w = torch.tensor(10,requires_grad=True,dtype=torch.float)
#定义损失函数loss
loss = 2*w**2
#求导
loss.backward()
#更新w
w.data = w.data-0.01*w.grad
#打印w(新)
print(w.data)