"""
采用torch.nn.MSELoss()函数作为loss函数
"""

import torch

torch.manual_seed(52)
#定义多项式中的x，采用torch.ones()函数自动生产全为1的张量
x=torch.ones((2,5))
#定义多项式中的y，采用torch.zeros()函数自动生产全为0的张量
y=torch.zeros((2,3))
print(f'张量x:{x}\n张量y:{y}')
#随机生成权重w
w=torch.randn(5,3,requires_grad=True)
#随机生成偏置项b
b=torch.randn(3,requires_grad=True)
print(f'权重w:{w}\n偏置项b:{b}')
#定义损失函数
z=torch.matmul(x,w)+b
loss=torch.nn.MSELoss()(z,y)

#求导
loss.backward()
#打印用来更新w和b的梯度
print(f'权重w:{w.grad}\n偏置项b:{b.grad}')
#更新w和b
w.data=w.data - 0.01 * w.grad
b.data=b.data - 0.01 * b.grad
print(f'权重w:{w}\n偏置项b:{b}')