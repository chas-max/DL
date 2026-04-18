"""
生成等差随机张量：
torch.arange()和torch.linspace()
设置随机种子
torch.initial_seed()和torch.manual_seed()
生成随机参数
torch.rand()和torch.randn()和torch.randint()
"""
import torch
def dm01():
    t1=torch.arange(1,10,2)#参数：开始值，结束值，步长
    print(f'张量：{t1},type:{type(t1)}')
    t2=torch.linspace(1,10,steps=3)
    print(f'张量：{t2},type:{type(t2)}')#参数：开始值，结束值，数值个数
    print('*'*52)
def dm02():
    # torch.initial_seed()
    torch.manual_seed(52)
    t3=torch.rand((2,3))
    print(f'张量：{t3},type:{type(t3)}')
    t4=torch.randn((2,3))
    print(f'张量：{t4},type:{type(t4)}')
    t5=torch.randint(1,20,(4,5))
    print(f'张量：{t5},type:{type(t5)}')
if __name__=='__main__':
    dm02()