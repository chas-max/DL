"""
形状转行中的API:
reshape()不改变数据内容
squeeze()增加一个维度，但不可越界,unsqueeze()删除为1的维度，不改变数据内容
transpose(),permute()改变数据内容
view()处理张量的要求:张量的数据顺序 和 内存中存储的顺序一致,contiguous()将顺序不一致的数据按照张量数据的顺序修改内存中的位置以使其一致,
is_contiguous()判断顺序是否一致
"""

import torch
torch.manual_seed(52)
def dm01():
    t1 = torch.randint(1,13,(2,3))
    print(f't1:{t1},t1.shape:{t1.shape[0]},column:{t1.shape[1]}')
    t2=t1.reshape(3,2)
    t3=t1.reshape(1,6)
    t4=t1.reshape(6,1)
    print(f't2:{t2},t2.shape:{t2.shape},column:{t2.shape[1]}')
    print(f't3:{t3},t3.shape:{t3.shape},column:{t3.shape[1]}')
    print(f't4:{t4},t4.shape:{t4.shape},column:{t4.shape[1]}')
def dm02():
    #unsqueeze增添一个维度，但不可越界
    t1=torch.randint(1,14,(2,3))
    print(f't1:{t1},t1.shape:{t1.shape[0]}')
    t2=t1.unsqueeze(0)
    print(f't2:{t2},t2.shape:{t2.shape}')
    t3=t1.unsqueeze(1)
    t4=t1.unsqueeze(2)
    print(f't3:{t3},t3.shape:{t3.shape}')
    print(f't4:{t4},t4.shape:{t4.shape}')
    #squeeze删除为1的维度
    t5=torch.randint(1,52,(2,1,3,1,1))
    print(f't5:{t5},t5.shape:{t5.shape}')
    t6=t5.squeeze()
    print(f't6:{t6},t6.shape:{t6.shape}')
def dm03():
    #transpose()一次仅可改变两个维度的顺序
    t1=torch.randint(1,13,(2,3,4))
    print(f'张量:{t1},形状:{t1.shape}')
    #要求1：(2,3，4)->(3,2,4)
    t2=t1.transpose(0,1)
    print(f'张量:{t2},形状:{t2.shape}')
    #要求2：(2,3,4)->(4,3,2)
    t3=t1.transpose(0,2)
    print(f'张量:{t3},形状:{t3.shape}')
    #permute()一次可改变多个维的顺序
    #场景1:(2,3,4)->(4,2,3)
    t4=t1.permute(2,0,1)
    print(f'张量:{t4},形状:{t4.shape}')
def dm04():
    #view()处理张量的要求:张量的数据顺序 和 内存中存储的顺序一致
    t1=torch.randint(1,13,(2,3))
    print(f'张量:{t1},形状:{t1.shape}')
    t2=t1.view(3,2)
    print(f'张量:{t2},形状:{t2.shape}')
    #使用reshape()改变张量的形状
    t3=t1.reshape(3,2)
    print(t3.is_contiguous())
    #使用transpose()改变张量的形状
    t4=t1.transpose(1,0)
    print(t4.is_contiguous())
    # t5=t3.view(6,1)
    # print(t5)
    t4 = t4.contiguous()
    print(t4.is_contiguous())
    t6 = t4.view(6,1)
    print(t6)
if __name__ == "__main__":
    # dm01()
    dm02()
    # dm03()
    # dm04()