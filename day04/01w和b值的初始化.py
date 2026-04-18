"""
初始化的目的：
防止数据爆炸和数据失真
提高数据的收敛率
破坏数据的对称性
数据初始化的方法：
随机生成(uniform_),正太分布(normal_),全零(zeros_),全一(ones_),固定值(constant_),kaiming_,xavier_
可以破坏对称的：
随机生成，正太分布，kaiming_,xavier_
常用zeros_初始化偏置,kaiming_+reLu,vaxier+tanh/sigmoid
随机生成使用于浅层神经网络,kaiming_,xavier_适用于深层神经网络
"""

from torch import nn
#随机生成 权重w 和 偏置b
def dm01():
    #创建线性层
    linear = nn.Linear(3,4)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)
#固定全零 权重w 和 偏置b
def dm02():
    # 创建线性层
    linear = nn.Linear(3, 4)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)
# 固定全一 权重w 和 偏置b
def dm03():
    # 创建线性层
    linear = nn.Linear(3, 4)
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)
# 固定值 权重w 和 偏置b
def dm04():
    # 创建线性层
    linear = nn.Linear(3, 4)
    #                 线性层权重         固定值
    nn.init.constant_(linear.weight,3.4)
    nn.init.constant_(linear.bias,3.4)
    print(linear.weight.data)
    print(linear.bias.data)
# 正态分布随机生成 权重w 和 偏置b
def dm05():
    # 创建线性层
    linear = nn.Linear(3, 4)
    #可以设置均值和标准差
    nn.init.normal_(linear.weight)
    nn.init.normal_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)
def dm06():
    #kaiming正态分布初始化，适用于深层神经网络std = sqrt(2 / fan_in),fan_in为上一输入层的神经元个数
    linear = nn.Linear(3,4)
    #kaiming_初始化对象必须为二维的，因此无法对linear.bias进行初始化
    # nn.init.kaiming_normal_(linear.weight)
    #kaiming_均匀分布初始化，适用于深层神经网络limit = sqrt(6 / fan_in),fan_in为上一输入层的神经元个数
    #数据范围为[-limit, limit]
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight.data)
def dm07():
    # kaiming正态分布初始化，适用于深层神经网络std = sqrt(2 / fan_in + fan_out),fan_in为上一输入层的神经元个数,fan_out为下一输入层的神经元个数
    linear = nn.Linear(3, 4)
    # xavier_初始化对象必须为二维的，因此无法对linear.bias进行初始化
    nn.init.xavier_normal_(linear.weight)
    # xavier_均匀分布初始化，适用于深层神经网络limit = sqrt(6 / fan_in + fan_out),fan_in为上一输入层的神经元个数,fan_out为下一输入层的神经元个数
    #数据范围为[-limit, limit]
    # nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    # dm06()
    dm07()