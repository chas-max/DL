"""
torch.zeros()和torch.zeros_like()创建填充0的张量
torch.ones()和torch.ones_like()创建填充1的张量
torch.full()和torch.full_like()创建填充指定值的张量
"""
import numpy as np
import torch

#场景1:torch.zeros()和torch.zeros_like()创建填充0的张量
t1=torch.zeros(2,3)
print(f'张量:{t1},type:{type(t1)}')
s=np.random.randint(1,6,size=(2,3))
t2=torch.tensor(s)
t3=torch.zeros_like(t2)
print(f'张量:{t3},type:{type(t3)}')
print('*'*52)
#场景2:torch.ones()和torch.ones_like()创建填充1的张量
t1=torch.ones(2,3)
print(f'张量:{t1},type:{type(t1)}')
s=np.random.randint(1,6,size=(2,3))
t2=torch.tensor(s)
t3=torch.ones_like(t2)
print(f'张量:{t3},type:{type(t3)}')
#场景3:torch.full()和torch.full_like()创建填充指定值的张量
t1=torch.full(size=(2,3),fill_value=255)
print('*'*52)
print(f'张量:{t1},type:{type(t1)}')
s=np.random.randint(1,6,size=(2,3))
t2=torch.tensor(s)
t3=torch.full_like(t2,fill_value=255)
print(f'张量:{t3},type:{type(t3)}')
print('*'*52)