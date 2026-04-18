"""
进行100次循环求loss值最小值以及对应的梯度
需要使用w.grad.zero_使梯度清零，因为grad属性会自动进行累加操作
"""

#导包
import torch

#定义初始w(旧)
w = torch.tensor(10,requires_grad=True,dtype=torch.float)
#定义损失函数
loss = w**2 + 20
for i in range(1,101):
    loss = w**2 + 20    #引入loss值，否则loss值不会进行更新
    if w.grad is not None:
        print(f'第{i}次循环，w值为：{w:.3f},loss值为：{loss:.3f},梯度为：{w.grad:.3f}')
        w.grad.zero_()  #梯度清零
    loss.backward()     # 反向传播
    w.data=w.data-0.01*w.grad   #更新w参数
print(f'w值为：{w:.3f},loss值为：{loss:.3f},梯度为：{w.grad:.3f}')