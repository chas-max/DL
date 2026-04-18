"""
L1正则化: 权重变为0
L2正则化: 权重无限趋近于0
随机失活Dropout: 使部分权重按照一定的概率p权重变为0，未失活的权重对其进行缩放，缩放比例为1/(1-p)
"""
#导包
import torch
import torch.nn as nn

#创建输入特征
x = torch.randint(1,10,size=(1,4),dtype = torch.float)
print(x)
#创建线性层
linear1 = nn.Linear(4,5)
#得到输出
output = linear1(x)
print(output)
s1= torch.relu(output)
print(s1)
dropout = nn.Dropout(p = 0.5)
d1 = dropout(s1)
print(d1)