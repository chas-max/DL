"""
张量转换为numpy：
张量.numpy()
numpy转化为张量：
torch.from_numpy(矩阵)
标量张量和数值转换
张量.item()
"""
import numpy as np
import torch


def dm01():
    t1=torch.tensor([1,2,3,4,5])
    print(f'张量：{t1},type:{type(t1)}')
    # n1=t1.numpy()           #共享内存
    n1=t1.numpy().copy()    #不共享内存
    print(f'numpy：{n1},type:{type(n1)}')    #张量->numpy
    n1[1]=100
    print(f'numpy：{n1},type:{type(n1)}')
    print(f'张量：{t1},type:{type(t1)}')
def dm02():
    n1=np.array([1,2,3,4,5])
    t1=torch.from_numpy(n1)     #共享内存
    t2=torch.tensor(n1)
    n1[1]=100
    print(f'numpy：{n1},type:{type(n1)}')
    print(f'张量：{t1},type:{type(t1)}')
    print(f'张量：{t2},type:{type(t2)}')
def dm03():
    t1=torch.tensor(1)
    print(f'张量：{t1},type:{type(t1)}')
    t2=t1.item()
    print(f'标量：{t2},type:{type(t2)}')
if __name__=='__main__':
    # dm01()
    # dm02()
    dm03()