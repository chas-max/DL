"""
CNN卷积神经网络：
    卷积层：提取图片特征
    池化层：降维，减小数据运算量，增强神经网络的鲁棒性
    全连接层
"""
import numpy as np
#导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#读取图片
img = plt.imread("preview.jpg")
#将图片转化为张量
img2 = torch.tensor(img).float()
#转换维度 将HWC->CHW，并增加一个维度以适应卷积层输入
img2 = img2.permute(2,0,1).unsqueeze(dim=0)
#定义卷积网络对象      输入通道数， 输出通道数，    卷积核大小， 步长，   填充
conv2d = nn.Conv2d(3,4,3,1,0)
img3 = conv2d(img2)
# print(f'img:{img},shape:{img.shape},type:{type(img)}')
#将图片降维后将CHW->HWC
img3 = img3.squeeze().permute(1,2,0)
#将张量转化为数组，以适应matplotlib打印数据要求
img3 = img3.detach().numpy()
plt.imshow(img3[:,:,3])
print(f'shape:{img3[0].shape}')
plt.show()