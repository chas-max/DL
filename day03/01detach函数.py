"""
numpy无法对可自行微分的张量进行转换，因此需要采用detach()函数进行拷贝后再转换
拷贝后的张量与原张量共享地址
"""
import torch
import numpy as np

t1 = torch.tensor([1,2,3,4],requires_grad=True,dtype=torch.float)
t2=t1.detach().numpy()
print(f'张量：{t1},种类：{type(t1)}')
print(f'张量：{t2},种类：{type(t2)}')
t1.data[0]=100
print(f'张量：{t1},种类：{type(t1)}')     #共享内存
print(f'张量：{t2},种类：{type(t2)}')