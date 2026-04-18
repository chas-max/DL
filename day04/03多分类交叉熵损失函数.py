"""
损失函数别名：目标函数 成本函数 代价函数 差错函数
多分类交叉熵损失函数(CrossEntropyLoss)用于拥有多个输出的神经网络的损失(拟合情况)的判断
多分类交叉熵损失函数(CrossEntropyLoss) = softmax() + 损失判断 因此在搭建神经网络的输出层时无需使用softmax()激活函数
公式:
loss = -∑ylog(S(f(x))
        x:输入特征
        f(x):线性求和
        S:softmax()求得概率
多分类交叉熵损失函数(CrossEntropyLoss)的目的是 求输入特征概率的对数的最小值
"""

import torch
from torch import nn
#定义真实值(y)
y_true = torch.tensor([[0,0,1],[1,0,0]],dtype = torch.float)
#真实值的另一种表达形式
# y_true = torch.tensor([2,0],dtype = torch.float)
y_pred = torch.tensor([[0.1,0.2,0.7],[0.8,0.1,0.1]],requires_grad=True,dtype = torch.float)
creterion = nn.CrossEntropyLoss()
loss = creterion(y_pred,y_true)
print(loss)