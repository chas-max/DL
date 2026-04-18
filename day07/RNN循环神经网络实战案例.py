"""
RNN循环神经网络搭建流程:
    1.准备数据集
"""
import time

import jieba
import torch.utils.data
from sympy.matrices.expressions.kronecker import explicit_kronecker_product
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def create_dataset():
    total_words, unique_words, words_count =[], [], 0
    # 读取数据集
    for line in open("./data/jaychou_lyrics.txt", "r", encoding="utf-8"):
        words = jieba.lcut(line)
        total_words.append(words)
    for words in total_words:
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    word_count = len(unique_words)
    # print("unique_words:", len(unique_words))
    word_to_idx = {word:i for i,word in enumerate(unique_words)}
    carpas_idx = []
    for words in total_words:
        tmp = []
        for word in words:
            tmp.append(word_to_idx[word])
        carpas_idx.extend(tmp)
        carpas_idx.append(word_to_idx[' '])
    return total_words, unique_words, word_count, carpas_idx
class lyricsmodel(torch.utils.data.Dataset):
        def __init__(self, char_num, carpas_idx):
            super().__init__()
            self.char_num = char_num
            self.carpas_idx = carpas_idx
            self.word_count = len(carpas_idx)
            #定义规定传入参数个数后的总词数
            self.number = self.word_count // self.char_num + 1
        #重新定义数据集长度
        def __len__(self):
            return self.number
        #重新定义数据集的索引
        def __getitem__(self,start):
            #对start进行限制，保证其在数据集长度内
            start = min(max(start, 0),self.word_count - self.char_num - 1)
            x = self.carpas_idx[start:start + self.char_num]
            y = self.carpas_idx[start + 1:start + self.char_num + 1]
            return torch.tensor(x),torch.tensor(y)
class RNN(nn.Module):
    def __init__(self,word_count):
        super().__init__()
        # 词嵌入层,参数           传入词的个数,    词向量的维度
        self.emd = nn.Embedding(word_count, 128)
        # RNN层，参数      输入的维度,   输出的维度,   隐藏层个数
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层，参数      输入的维度,   输出的维度（转化为每个词对应的概率）
        self.out = nn.Linear(256, word_count)
    def forward(self,input, hidden):
        embd = self.emd(input)
        #transpose将(batch_size, seq_len, hidden_size)变为(seq_len, batch_size, hidden_size)
        output, hidden = self.rnn(embd.transpose(0,1), hidden)
        #reshape将(seq_len, batch_size, hidden_size)变为(batch_size, seq_len, hidden_size)
        output = self.out(output.reshape(-1,output.shape[-1]))
        return output, hidden
    def init_hidden(self,bs):
        # 初始化隐藏层
        return torch.zeros(1 ,bs, 256)
def model_train():
    total_words, unique_words, word_count, carpas_idx = create_dataset()
    #创建模型
    model = RNN(word_count)
    #创建数据集
    dataset = lyricsmodel(32, carpas_idx)
    #创建数据加载器
    train_loader = DataLoader(dataset, 5, shuffle = True)
    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3,betas=(0.9, 0.999))
    epochs = 10
    for epoch in range(epochs):
        start, total_loss, total_sampels = time.time(), 0.0, 0
        for x,y in train_loader:
            hidden = model.init_hidden(x.shape[0])
            output, hidden = model(x, hidden)
            #y的维度是(batch_size, seq_len, hidden_size)，需要将其变为(batch_size * seq_len, hidden_size)
            y = y.transpose(0,1).reshape(-1,)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sampels += 1
        #本轮训练完成，输出损失值和训练时间
        print(f'轮数:{epoch+1},损失值为:{total_loss/total_sampels},时间为:{time.time()-start}s')
    #模型训练完成，保存模型
    torch.save(model.state_dict(),'./model/textmodel.pth')
def model_evaluate(start_dict, length):
    total_words, unique_words, word_count, carpas_idx = create_dataset()
    word_to_idx = {word:i for i,word in enumerate(unique_words)}
    model = RNN(word_count)
    model.load_state_dict(torch.load('./model/textmodel.pth'))
    dataset = lyricsmodel(32, carpas_idx)
    hidden = model.init_hidden(1)
    for i in range(length):
        start_idx = word_to_idx[start_dict]
        output, hidden = model(torch.tensor([[start_idx]]), hidden)
        generate_txt = torch.argmax(output).item()
        start_dict = unique_words[generate_txt]
        print(start_dict, end='')

if __name__ == "__main__":
    # print("total_words:", len(total_words))
    # print("unique_words:", unique_words)
    # print("word_count:", word_count)
    # print("carpas_idx:", carpas_idx)
    # dataset = lyricsmodel(5, carpas_idx)
    # print(len(dataset))
    # x, y = dataset[1]
    # print(f'输入值{x}')
    # print(f'输出值{y}')# Python
    # model = RNN(word_count)
    # print(model)
    model_evaluate('星星',50)