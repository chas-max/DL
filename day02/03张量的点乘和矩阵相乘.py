"""
张量的点乘：
张量.mul(张量)，t1*t2          张量的形状必须一致
矩阵的相乘
t1@t2和t1.matmul(t2)
要求：
t1行=t2列
"""
import torch
def dm01():
    t1=torch.tensor([[2,3,4],[5,6,7]])
    t2=torch.tensor([[2,3,4],[2,3,4]])
    t3=t1*t2
    t4=t1.mul(t2)
    print(f"张量结果为：{t3}")
    print(f"张量结果为：{t4}")
def dm02():
    # t1=torch.tensor([[2,3,4],[5,6,7]])
    # # t2=torch.tensor([[1,2],[3,4],[5,6]])
    # t3=t1@t2
    # t4=t1.matmul(t2)
    # print(f"张量结果为：{t3}")
    # print(f"张量结果为：{t4}")
    t1=torch.tensor([2,3,4])
    t2=torch.tensor([2,3,4])
    t5=t1.dot(t2)
    print(f"张量结果为：{t5}")
if __name__=="__main__":
    # dm01()
    dm02()