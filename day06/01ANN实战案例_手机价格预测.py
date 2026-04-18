"""
手机价格预测实战案例程序流程：
    1. 准备数据
    2. 搭建神经网络
    3. 训练模型
    4. 测试模型
"""

#导包
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import time


#创建数据集
def create_dataset():
    dataset = pd.read_csv("手机价格预测.csv")
    # print(dataset)
    #转化数据类型
    dataset = dataset.astype(np.float32)
    #数据特征和数据标签
    x = dataset.iloc[:,:-1]
    y = dataset.iloc[:,-1]
    #进行数据划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,stratify = y, random_state = 52)
    #准备数据加载器DataLoader,数据转化流程 numpy->tensor->数据集TensorDataset->数据加载器DataLoader
    # print(x_train, x_train.shape)
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    # print(train_dataset)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y_train))
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim,128)
        self.linear2 = nn.Linear(128,256)
        self.output = nn.Linear(256,output_dim)
    def forward(self,x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        output = self.output(x)
        return output
def train_model(train_dataset, input_dim, output_dim):
    #创建数据加载器DataLoader
    train_loader  = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    #创建模型
    model = PhonePriceModel(input_dim, output_dim)
    #创建损失函数
    criterion = nn.CrossEntropyLoss()
    #创建优化器
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss, batch_num = 0.0,0
        start_time = time.time()
        for x,y in train_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print("epoch:{}, loss:{:.4f}, time:{:.4f}".format(epoch, total_loss/batch_num, time.time()-start_time))
    torch.save(model.state_dict(),'model/手机价格预测.pth')
def test_model(test_dataset, input_dim, output_dim):
    #创建数据加载器DataLoader
    test_loader  = DataLoader(test_dataset, batch_size = 8, shuffle = False)
    #下载模型
    model = PhonePriceModel(input_dim, output_dim)
    model.load_state_dict(torch.load('model/手机价格预测.pth', weights_only = False))
    correct = 0 #记录正确预测的数量
    model.eval()
    for x,y in test_loader:
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim = 1)
        print(y_pred)
        correct += (y_pred==y).sum().item()
        print(f'正确率(Accuracy):{correct/len(test_dataset)}')
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim =create_dataset()
    # model = PhonePriceModel(input_dim, output_dim)
    # summary(model, (16, input_dim))
    train_model(train_dataset, input_dim, output_dim)
    test_model(test_dataset, input_dim, output_dim)