"""
CNN卷积神经网络:词嵌入层, 循环网络层, 输出层
常用来处理存在先后顺序的数据 序列数据 ,例如,情感分析, 机器翻译, 文本分类, 翻译, 主题分析, 提取关键信息
词嵌入层API:Embedding()     作用:将对应的词按照对应的词 (或词对应的索引) 转化为词向量
"""

import torch
import jieba
from torch.nn import Embedding

txt = '北京冬奥的进度已经过半，不少外国运动员在完成自己的比赛后踏上归程。'
#运用jieba库函数分词
words = jieba.lcut(txt)
print(words)
#每次生成的词向量的数字不同
#参数:            输入的词的个数 , 词向量的维度
embid = Embedding(len(words),4)
for i,word in enumerate(words):
    print(f'{i}--{word}--{embid(torch.tensor(i))}')#Embedding()接受的数据形式为张量,因此需将其转化为张量形式