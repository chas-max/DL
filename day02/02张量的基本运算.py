"""
张量的基本运算：
add/sub/mul/div ==+/-/*// neg取反
add_/sub_/mul_/div_==+=/-=/*=//= neg_取反 基本功能同上，
但是修改源变量,相当于inplace=True
"""
import torch
t1=torch.tensor([1,2,3,4])
# t2=t1.add(10)
# t1.add_(10)
# t2=t1+10
# t1+=10
# t2=t1.sub(1)
# t3=t1.mul(2)
# t4=t1.div(2)
t5=t1.neg()
# print(f'{t1},type:{type(t1)}')
# print(f'{t2},type:{type(t2)}')
# print(f'{t3},type:{type(t3)}')
# print(f'{t4},type:{type(t4)}')
print(f'{t5},type:{type(t5)}')