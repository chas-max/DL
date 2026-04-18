"""
常见的索引操作有：
普通行列索引
列表索引
范围索引
布尔索引
多维索引
"""
import torch

torch.manual_seed(42)
t1=torch.randint(1,10,size=(5,5))
print(f'张量t1:{t1}')
# #普通索引
# print(t1[[1,4],[2,3]])
# #列表索引
# print(t1[:,[2,3]])
# print(t1[2:3,:])
# #范围索引
print(t1[[[1],[2]],[2,3]])
# print(t1[1:2,2:4])
# print(t1[::2,1::2])
# #布尔索引
# print(t1[1][torch.tensor([True,False,True,False,True])])
# print(t1[t1>5])
# print(t1[:3][t1[:3]>5])

# t2=torch.randint(1,10,(2,3,4))
# print(t2)
# print(t2[:,:,0])
# print(t2[0,:,:])
# print(t2[:,0,:])