"""
深度学习的四个步骤：
生成数据
搭建神经网络
训练神经网络
测试神经网络
搭建神经网络：
继承nn.Module类
初始化init生成网络
前向传播(forward)
"""
import torch
from torch import nn
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        #初始化父类
        super().__init__()
        #定义线性层(输入层神经元3个,输出层神经元3个)
        self.linear1 = nn.Linear(3,3)
        #定义线性层(输入层神经元3个,输出层神经元2个)
        self.linear2 = nn.Linear(3,2,)
        #定义线性层(输入层神经元2个,输出层神经元2个)
        self.output = nn.Linear(2,2)
        #初始化 权重w 和 偏置b,设置权重有 kaiming_ 和 xavier_ 随机生成,偏置为0
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        #自动调用forward,因此不可更改方法名
    def forward(self,x):
        #第一层 采用xavier_进行权重w的初始化和采用sigmoid激活函数进行非线性化
        x=torch.sigmoid(self.linear1(x))
        #第一层 采用kaiming_进行权重w的初始化和采用reLu激活函数进行非线性化
        x=torch.relu(self.linear2(x))
        #第一层 采用xavier_进行权重w的初始化和采用softmax激活函数进行非线性化,dim=-1为最后一维进行归一化
        x=torch.softmax(self.output(x),dim=-1)
        return x

def train():
    #生成数据
    data = torch.randn(6,3)
    model = Net()
    output=model(data)
    print(f'{data}\n')
    #自动调用forward函数,使output自动求导
    # print(f'data.requires_grad:{output.requires_grad}')
    # summary(model,(6,3))
    print('='*52)
    #打印初始的 权重 和 偏置
    for name,para in model.named_parameters():
        print(name)
        print(para,'\n')
    print('='*52)
if __name__ == '__main__':
    train()