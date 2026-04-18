"""
在机器学习中的数据均采用张量的形式，同时张量要求为浮点数
torch.tensor:可以将标量，列表，举证转化为张量
torch.Tensor：在torch.tensor的基础上增加了可以修改张量的形状
torch.IntTensor,torch.FloatTensor：指定张量的数据类型
"""
import torch
import numpy as np
#demo1
def demo1():
    s=torch.tensor(10,dtype=float)
    print(f'张量：{s},type:{type(s)}')
    lst=[1,2,3]
    s=torch.tensor(lst)
    print(f"张量：{s},type:{type(s)}")
    j=np.random.randint(1,10,size=(2,5))
    s=torch.tensor(j)
    print(f"张量：{s},type:{type(s)}")

#demo2
def demo2():
    s=torch.Tensor(10)
    print(f'张量：{s},type:{type(s)}')
    lst=[1,2,3]
    s=torch.Tensor(lst)
    print(f"张量：{s},type:{type(s)}")
    j=np.random.randint(1,10,size=(2,5))
    s=torch.Tensor(j)
    print(f"张量：{s},type:{type(s)}")
    s=torch.Tensor(2,5)
    print(f"张量:{s},type:{type(s)}")
#demo3
def demo3():
    s=torch.FloatTensor(10)
    print(f'张量：{s},type:{type(s)}')
    lst=[1,2,3]
    s=torch.FloatTensor(lst)
    print(f"张量：{s},type:{type(s)}")
    j=np.random.randint(1,10,size=(2,5))
    s=torch.FloatTensor(j)
    print(f"张量：{s},type:{type(s)}")
    s=torch.FloatTensor(2,5)
    print(f"张量:{s},type:{type(s)}")
if __name__=="__main__":
    demo1()