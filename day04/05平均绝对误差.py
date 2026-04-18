"""
平均绝对误差公式(MAE) L1Loss():
        loss = 1/n*∑|y_pred-y_true|
    n:为预测值或真实值的个数
常作为其他公式的惩罚公式,L1正则化
由于0处不可导,因此可以跳过极小值

均方误差(MSE) MSELoss():
        loss = 1/n*∑(y_pred - y_true)**2
    n:为预测值或真实值的个数
常用作L2正则化
如果参数较大可能出现梯度爆炸的情况

smooth L1误差公式 SmoothL1Loss():
        loss = 0.5**2   if |x|<1
        loss = |x| - 0.5 if |x|>=1
smooth L1综合 L1(MAE) 和 L2(MSE) 的优点解决了L1在0点不平滑(梯度消失)的问题以及L2梯度爆炸的问题
"""

#导包
import torch
import torch.nn as nn
def dm01():
    #定义真实值,要求为 torch.float32 类型
    y_true = torch.tensor([2, 1.9, 1.5],dtype =torch.float)
    #定义预测值,要求为 torch.float32 类型
    y_pred = torch.tensor([2.1, 1.8, 1.6])
    #定义平均绝对误差MAE对象
    criterion = nn.L1Loss()
    #计算平均绝对误差
    loss = criterion(y_pred,y_true)
    print(f'损失值:{loss.data}')
def dm02():
    #定义真实值,要求为 torch.float32 类型
    y_true = torch.tensor([2, 1.9, 1.5],dtype =torch.float)
    #定义预测值,要求为 torch.float32 类型
    y_pred = torch.tensor([2.1, 1.8, 1.6])
    #定义平均绝对误差MSELoss对象
    criterion = nn.MSELoss()
    #计算平均绝对误差
    loss = criterion(y_pred,y_true)
    print(f'损失值:{loss.data}')
def dm03():
    #定义真实值,要求为 torch.float32 类型
    y_true = torch.tensor([2, 1.9, 1.5],dtype =torch.float)
    #定义预测值,要求为 torch.float32 类型
    y_pred = torch.tensor([2.1, 1.8, 1.6])
    #定义平均绝对误差SmoothL1Loss对象
    criterion = nn.SmoothL1Loss()
    #计算平均绝对误差
    loss = criterion(y_pred,y_true)
    print(f'损失值:{loss.data}')
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()