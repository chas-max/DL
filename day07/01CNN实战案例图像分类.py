"""
CNN实战案例:
    运用torchvision中包含的50000张训练集，10000张测试集对模型进行训练
    实战流程:
        1.准备数据
"""
import time

from torch.optim import Adam
from torch.utils.data import DataLoader

BATCH_SIZE = 8
import torch
from torch import nn
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def create_dataset():
    train_dataset = CIFAR10('./data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10('./data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1,0)
        self.pool1 = nn.MaxPool2d(2,2,0)
        self.conv2 = nn.Conv2d(6,16,3,1,0)
        self.pool2 = nn.MaxPool2d(2,2,0)
        self.linear1= nn.Linear(576,120)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(120,84)
        self.output = nn.Linear(84,10)
    def forward(self,x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0),-1)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        return self.output(x)#后续损失函数的计算采用CrossEntropyLoss()
def model_train(model,train_dataset):
    #创建数据加载器
    dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE,shuffle=True)
    #创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    #创建优化器对象
    optimizer = Adam(model.parameters(),lr=1e-3)
    #定义训练轮数
    epochs = 10
    #切换模型为训练模式
    model.train()
    for epoch in range(epochs):
        #训练总损失，训练总次数，训练正确数，训练开始时间
        total_loss, loss_samples, total_crrects, start_time = 0.0, 0, 0, time.time()
        for x,y in dataloader:
            #模型前向传播，数据预测
            y_pred = model(x)
            #计算损失值
            loss = criterion(y_pred,y.long())
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #优化器更新参数，前向传播
            optimizer.step()
            total_loss += loss.item()*len(y)
            loss_samples += len(y)
            #torch.argmax()获取经过预测概率最大值的索引,经过 == 判断是否相同,求和得到预测正确数
            total_crrects += (torch.argmax(y_pred,dim=-1) == y).sum()
        print(f'epoch{epoch+1},loss:{total_loss/loss_samples:.3f},crrects:{total_crrects/loss_samples:.3f},time:{time.time()-start_time:.3f}')
    torch.save(model.state_dict(),'./model/图像分类.pth')
def model_test(test_dataset):
    #创建数据加载器
    dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    #加载模型
    model = CNN()
    model.load_state_dict(torch.load('./model/图像分类.pth'))
    #切换模型为测试模式
    model.eval()
    total_coerrcts, loss_samples = 0,0
    for x,y in dataloader:
        #数据预测，但由于在搭建模型时候未使用softmax,因此需要进行后续操作方可的到对于的种类
        y_pred =model(x)
        #torch.argmax()获取经过预测概率最大值的索引,经过 == 判断是否相同,求和得到预测正确数
        total_coerrcts += (torch.argmax(y_pred,dim=-1) == y).sum()
        #统计数据样本数
        loss_samples += len(y)
    print(f'测试集准确率:{total_coerrcts/loss_samples:.3f}')
if __name__== '__main__':
    train_dataset, test_dataset = create_dataset()
    # print(f'shape:{train_dataset.data.shape}')
    model = CNN()
    # summary(model,(3,32,32),batch_size=1)
    model_train(model,train_dataset)
    model_test(test_dataset)