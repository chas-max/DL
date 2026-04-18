"""
图像分类：二值图，灰度图，索引图，RBG图
常用的API:
    imshow()
    imread()
    imsave()
"""
import torch
import numpy
import matplotlib.pyplot as plt

def dm01():
    s1 = torch.full((255,255,3), 255)
    plt.imshow(s1)
    plt.show()
    s2 = torch.zeros(255,255,3)
    plt.imshow(s2)
    plt.axis('off')
    plt.show()
def dm02():
    img = plt.imread("preview.jpg")
    print(f'img:{img},shape:{img.shape}')
if __name__ == '__main__':
    # dm01()
    dm02()