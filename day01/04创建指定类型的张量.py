"""
张量类型的转换
type(torch支持的数据类型)
.half()/.float()/.doubel()/.short()/.int()/.long()
"""
import torch
t1=torch.tensor([1,2,3,4,5],dtype=torch.int)
print(f'张量：{t1},元素类型{t1.dtype},张量类型{type(t1)}')
t2=t1.type(torch.float)
print(f'张量：{t2},元素类型{t2.dtype},张量类型{type(t2)}')
print('*'*52)
# .half()/.float()/.double()/.short()/.int()/.long()
t3=t1.half()#float16
t4=t1.float()#float32
t5=t1.double()#float64
t6=t1.short()#int16
t7=t1.int()#int32
t8=t1.long()#int64
print(f'张量：{t3},元素类型{t3.dtype},张量类型{type(t3)}')
print(f'张量：{t4},元素类型{t4.dtype},张量类型{type(t4)}')
print(f'张量：{t5},元素类型{t5.dtype},张量类型{type(t5)}')
print(f'张量：{t6},元素类型{t6.dtype},张量类型{type(t6)}')
print(f'张量：{t7},元素类型{t7.dtype},张量类型{type(t7)}')
print(f'张量：{t8},元素类型{t8.dtype},张量类型{type(t8)}')
