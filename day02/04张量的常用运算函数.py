"""
张量的常用API:
mean(),max(),min(),sum()        有dim参数,dim同numpy中的axes
sqrt(),pow(),exp(),log(),log2(),log10() 没有dim参数
"""
import torch

#max()取最大值dim=1为行取最大值,dim=0为列取最大值,不传参则取所有元素最大值
t1=torch.tensor([[2,3,4],[5,6,7]],dtype=float)
print(f"张量为：{t1}")
# t2=t1.max(dim=1)
# t3=t1.max(dim=0)
# t4=t1.max()
# print(f'张量最大值为：{t2}')
# print(f'张量最大值为：{t3}')
#min()取最小值dim=1为行取最大值,dim=0为列取最大值,不传参则取所有元素最大值
# t2=t1.min(dim=1)
# t3=t1.min(dim=0)
# t4=t1.min()
# print(f'张量最小值为：{t2}')
# print(f'张量最小值为：{t3}')
# print(f'张量最小值为：{t4}')
#mean()取最平均值dim=1为行取最大值,dim=0为列取最大值,不传参则取所有元素最大值
# t2=t1.mean(dim=1)
# t3=t1.mean(dim=0)
# t4=t1.mean()
# print(f'张量平均值为：{t2}')
# print(f'张量平均值为：{t3}')
# print(f'张量平均值为：{t4}')
#sum()取最和值dim=1为行取最大值,dim=0为列取最大值,不传参则取所有元素最大值
# t2=t1.sum(dim=1)
# t3=t1.sum(dim=0)
# t4=t1.sum()
# print(f'张量总和值为：{t2}')
# print(f'张量总和值为：{t3}')
# print(f'张量总和值为：{t4}')
#pow()取幂值
# t4=torch.pow(t1,2)
# print(f'张量平方值为：{t4}')
#pow()取开方值
# t4=torch.sqrt(t1)
# print(f'张量平方值为：{t4}')
#pow()取开方值
# t4=torch.exp(t1)
# print(f'张量平方值为：{t4}')
# t4=torch.log(t1)
# print(f'张量平方值为：{t4}')
t4=torch.log2(t1)
print(f'张量平方值为：{t4}')
# t4=torch.log10(t1)
# print(f'张量平方值为：{t4}')
