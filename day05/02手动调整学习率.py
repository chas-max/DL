"""
除RMSProp,AdaGrad,Adam可自动调整学习率外，我们还可以运行函数进行 等间隔，指定间隔，指数间隔进行学习率的调整
等间隔lr_scheduler.StepLR()函数
参数：
step_size:指定的调整步长
gamma: 学习率调整系数

指定间隔lr_sheduler.MultiStepLR()
milestones: 指定的调整学习率的轮次
gamma: 学习率调整系数

指数衰减:
    lr新 = lr旧 * gamma ** epoch
gamma: 学习率调整系数

等间隔俗衰减：
    优点：简单，适合大数据
    缺点：学习率调整过快，可能会跳过最小值点
    使用场景：大数据，较为简单的任务
指定间隔：
    优点：简单，平稳
    缺点：在某些情况下，可能学习率衰减过快，导致提前停止
    使用场景：适用要求平稳的数据
指数衰减：
    优点：平滑，考虑历史，平稳
    缺点：需要更多的资源
    使用场景：精度要求高，避免过快收敛
"""

#导包
import torch
from torch import optim
import matplotlib.pyplot as plt

def dm01():
    #数据初始化
    lr, epochs, iteration = 0.1, 200, 10
    x = torch.tensor(1.0, requires_grad = True)
    y_true = torch.tensor(2.0, dtype = torch.float)
    w = torch.tensor(1.0, requires_grad= True)

    #创建优化器
    optimizer = optim.SGD([w], lr = lr, momentum = 0.9)
    #定义学习率衰减对象
    #等间隔参数:                                 优化器      调整步长         衰减调整系数
    #lr(新）= lr(旧) * gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.5)
    #创建学习率和轮次列表
    lr_list, epoch_list = [], []
    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
        for i in range(iteration):
            y_pred = w * x
            #采用最小二乘法算损失函数
            loss = (y_pred - y_true)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    #绘制学习率变化曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.show()
def dm02():
    # 数据初始化
    lr, epochs, iteration = 0.1, 200, 10
    x = torch.tensor(1.0, requires_grad=True)
    y_true = torch.tensor(2.0, dtype=torch.float)
    w = torch.tensor(1.0, requires_grad=True)

    # 创建优化器
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 定义学习率衰减对象
    # ；指定间隔参数:                             优化器       指定的间隔                衰减调整系数
    # lr(新）= lr(旧) * gamma
    milestones = [50, 100 ,160]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=0.5)
    # 创建学习率和轮次列表
    lr_list, epoch_list = [], []
    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
        for i in range(iteration):
            y_pred = w * x
            # 采用最小二乘法算损失函数
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    # 绘制学习率变化曲线
    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.show()
def dm03():
    # 数据初始化
    lr, epochs, iteration = 0.1, 200, 10
    x = torch.tensor(1.0, requires_grad=True)
    y_true = torch.tensor(2.0, dtype=torch.float)
    w = torch.tensor(1.0, requires_grad=True)

    # 创建优化器
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 定义学习率衰减对象
    # ；指定间隔参数:                             优化器       指定的间隔                衰减调整系数
    # lr(新）= lr(旧) * gamma
    milestones = [50, 100 ,160]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=0.5)
    #示例3：学习率指数衰减
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # 创建学习率和轮次列表
    lr_list, epoch_list = [], []
    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())
        for i in range(iteration):
            y_pred = w * x
            # 采用最小二乘法算损失函数
            loss = (y_pred - y_true) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    # 绘制学习率变化曲线
    plt.plot(epoch_list, lr_list)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.show()
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()