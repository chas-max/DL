"""
张量拼接API:
cat()要求除了拼接的维度外，其他维度一致，不增加新维度，例如(2,3)+(3,3)->(5,3)
stack()要求所有维度的形状均一致，会增加新维度
"""
import torch
#cat()cat()要求除了拼接的维度外，其他维度一致，不增加新维度
def dm01():
    t1 = torch.randint(1,9,(3,3))
    print(f'张量t1:{t1},形状:{t1.shape}')
    t2 = torch.randint(1,9,(2,3))
    print(f'张量t2:{t2},形状:{t2.shape}')
    #行拼接
    # t3=torch.cat([t1,t2],dim=1)
    # print(f'张量t3:{t3},形状:{t3.shape}')
    #列拼接
    t4 = torch.cat([t1,t2],dim=0)
    print(f'张量t4:{t4},形状:{t4.shape}')
# stack()要求所有维度的形状均一致，会增加新维度
def dm02():
    t1=torch.randint(1,9,(2,3))
    print(f'张量t1:{t1},形状:{t1.shape}')
    t2=torch.randint(1,9,(2,3))
    print(f'张量t2:{t2},形状:{t2.shape}')
    t3=torch.stack([t1,t2],dim=0) #(2,2,3)
    print(f'张量t3:{t3},形状:{t3.shape}')
    t4=torch.stack([t1,t2],dim=1)
    print(f'张量t4:{t4},形状:{t4.shape}')
    t5=torch.stack([t1,t2],dim=2)
    print(f'张量t5{t5},形状:{t5.shape}')
if __name__ == '__main__':
    # dm01()
    dm02()