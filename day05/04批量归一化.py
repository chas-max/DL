"""
批量归一化：先对数据进行加权求和，然后对加权求和的结果进行模式化（公式：(s - u) / sqrt(σ**2+小常数）（失去一些数据）
最后对模式化后的数据进行 缩放 和 平移（找回一些数据）
"""
#导包
import torch
import torch.nn as nn
def dm01():
    #创建输入特征
    x = torch.randint(1,10,size=(1,2,3,4),dtype = torch.float)
    linear = nn.Linear(4,4)
    #加权求和
    output = linear(x)
    print(x)
    print(output)
    #创建二维批量归一化对象
    #参数：
        # 1.输入特征维度
        # 2.eps：防止除零
        # 3.momentum：统计量的移动平均参数
        # 4.affine：是否进行缩放平移
    bn2 = nn.BatchNorm2d(2, eps = 1e-5, momentum = 0.1, affine = True)
    result = bn2(output)
    print(result)
def dm02():
    #创建输入特征
    x = torch.randint(1,10,size=(3,4),dtype = torch.float)
    linear = nn.Linear(4,4)
    #加权求和
    output = linear(x)
    print(x)
    print(output)
    #创建二维批量归一化对象
    #参数：
        # 1.输入特征维度
        # 2.eps：防止除零
        # 3.momentum：统计量的移动平均参数
        # 4.affine：是否进行缩放平移
    bn2 = nn.BatchNorm1d(2, eps = 1e-5, momentum = 0.1, affine = True)
if __name__ == '__main__':
    # dm01()
    dm02()