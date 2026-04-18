"""
传统梯度更新方法:
        w(新) = w(旧) - n * grad
若遇到较平滑的地方会出现梯度更新缓慢的问题
若遇到 鞍点(梯度为0的点) 则无法更新梯度

动量法Momentum公式:
    St = β*St-1 + (1-β)Gt t>=1
         G1               t=0
    St: 本次指数移动加权平均
    β:  调节权重系数，β越大则w越平缓，反之，则越陡峭
    Gt: 本次的梯度(不考虑历史梯度)
梯度更新公式:
    w(新) = w(旧） - n * St
解决了平滑的地方会出现梯度更新缓慢的问题
解决了 鞍点(梯度为0的点) 则无法更新梯度

自适应学习率AdaGrad公式:
    St = St-1 +Gt * Gt
    St: 本次累计权重和
    St-1: 历史累计权重和
    Gt: 本次权重(不含历史权重)
学习更细公式:
    学习率 = n/(sqrt(St)+小常数)
权重更新公式:
    w(新) = w(旧) - n(更新后的学习率)*Gt
可能出现的问题:过早的使学习率过小，导致权重w的更新缓慢

自适应学习率RSMProp公式:
    St = β * St-1+ (1-β) * Gt * Gt
    St: 本次累计权重和
    St-1: 历史累计权重和
    Gt: 本次权重(不含历史权重)
    β: 权重调和系数
学习更细公式:
    学习率 = n/(sqrt(St)+小常数)
权重更新公式:
    w(新) = w(旧) - n(更新后的学习率)*Gt
优点:是对 AdaGrad 的优化，解决了可能出现的梯度下降缓慢的问题

自适应矩估计公式:
    Mt = β1 * Mt-1 + (1-β1) * Gt
    St = β2 * St-1 + (1-β2) *Gt * Gt
    Mt^ = Mt / 1-β^t
    St^ = St / 1-β^t
    w(新) = w(旧) - (n / (sqrt(St^) + 小常数)) * Mt^
自适应矩估计  = RMSProp + Monmentum
总结：梯度下降优化方法的选择：
    简单模型，小数据：
        SGD,Momnetum
    复杂模型，大数据：
        adam
    稀疏数据。文本处理：
        AdaGrad,RMSProp
"""

#导包
import torch
import torch.nn as nn
from torch.optim import SGD
def dm01_momentum():
    #定义初始权重w
    w = torch.tensor([1], requires_grad=True, dtype = torch.float)
    #定义损失函数
    criterion = 1/2*w**2
    loss = criterion.sum()
    #创建优化器
    optimizer = SGD([w], lr= 0.01, momentum=0.9)
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
    loss = (1/2*w**2).sum()
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
def dm02_AdaGrad():
    #定义初始权重w
    w = torch.tensor([1], requires_grad=True, dtype = torch.float)
    #定义损失函数
    criterion = 1/2*w**2
    loss = criterion.sum()
    #创建优化器
    # optimizer = SGD([w], lr= 0.01, momentum=0.9)
    optimizer = torch.optim.Adagrad([w], lr= 0.01)
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
    loss = (1/2*w**2).sum()
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
def dm03_rmsprop():
    #定义初始权重w
    w = torch.tensor([1], requires_grad=True, dtype = torch.float)
    #定义损失函数
    criterion = 1/2*w**2
    loss = criterion.sum()
    #创建优化器
    # optimizer = SGD([w], lr= 0.01, momentum=0.9)
    # optimizer = torch.optim.Adagrad([w], lr= 0.01)
    optimizer = torch.optim.RMSprop([w], lr=0.01, alpha = 0.99)
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
    loss = (1/2*w**2).sum()
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
def dm04_adam():
    #定义初始权重w
    w = torch.tensor([1], requires_grad=True, dtype = torch.float)
    #定义损失函数
    criterion = 1/2*w**2
    loss = criterion.sum()
    #创建优化器
    # optimizer = SGD([w], lr= 0.01, momentum=0.9)
    # optimizer = torch.optim.Adagrad([w], lr= 0.01)
    # optimizer = torch.optim.RMSprop([w], lr=0.01, alpha = 0.99)
    #梯度清零
    optimizer = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999))
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
    loss = (1/2*w**2).sum()
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    optimizer.step()
    print(f'w值为：{w.data[0]:.3f},梯度为：{w.grad.data[0]:.3f}')
if __name__ == '__main__':
    # dm01_momentum()
    # dm02_AdaGrad()
    # dm03_rmsprop()
    dm04_adam()