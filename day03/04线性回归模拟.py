import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from torch.nn import Linear, MSELoss
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dataset():
    x,y,coef=make_regression(
        n_samples = 100,  #样本数量
        n_features = 1,   #特征数量为1
        noise = 10,       #添加噪声
        coef = True,      #是否返回权重
        bias = 14.5,      #偏置
        random_state = 52
    )
    x = torch.from_numpy(x).float()
    y = torch.tensor(y,dtype = torch.float)
    return x,y,coef
def train(x,y,coef):
    #tensor张量->数据集Dataset
    dataset = TensorDataset(x,y)
    #数据集->数据加载器DataLoader，参数：数据集(Dataset),批次大小，是否将数据打乱
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    #创建线性模型
    model = Linear(1,1)
    #创建损失函数
    certerion = MSELoss()
    #创建优化器
    optimizer = SGD(model.parameters(),lr=0.01)
    #创建训练轮次，每轮(平均)损失值，总损失值，平均损失值
    epochs,loss_list,total_loss,samples_count = 100,[],0,0
    for i in range(1,epochs+1):
        for x_train,y_train in dataloader:
            y_pred = model(x_train)                             #预测值
            loss = certerion(y_pred,y_train.reshape(-1,1))      #计算损失值，要求预测值和真实值的形状相同
            total_loss += loss.item()
            samples_count += 1
            optimizer.zero_grad()                               #梯度清零
            loss.backward()                                     #反向传播
            optimizer.step()                                    #优化器更新参数
        loss_list.append(total_loss/samples_count)
        print(f'epoch:{i},loss:{total_loss/samples_count}')
    print(f'total_loss:{total_loss},avg_loss:{total_loss/epochs}')
    print(f'coef:{model.weight[0][0]},bias:{model.bias[0]}')
    return model.weight[0][0],model.bias[0],loss_list
#参数：x,y,loss_list,coef,weight,bias
def draw_graphics(x,y,coef,weight,bias,loss_list):
    #绘制损失值变化曲线
    plt.plot(range(100),loss_list)
    plt.title('损失值变化曲线')
    plt.grid()
    plt.show()
    #绘制x,y散点图，预测直线和实际直线
    plt.scatter(x,y)
    y_pred = torch.tensor([v * weight + bias for v in x])
    y_true = torch.tensor([v * coef + 14.5 for v in x])
    plt.plot(x,y_pred,color = 'yellow',label = '预测值')
    plt.plot(x,y_true,color = 'red',label = '实际值')
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == '__main__':
    x,y,coef=create_dataset()
    # print(f'x:{x}type:{x.dtype}\ny:{y},type:{y.dtype}\ncoef:{coef}')#创建数据集
    weight,bias,loss_list = train(x,y,coef)
    print(f'coef:{coef}')
    draw_graphics(x,y,coef,weight,bias,loss_list)
