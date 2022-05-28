# Pytorch基础教程与常用操作

本教程基于本人常用的pytorch操作进行整理，用于帮助快速入门与某些torch技巧的查阅，同时能帮助我们看懂别人的代码，而非涉及pytorch的所有内容，仅限参考。

深入学习请多花时间阅读pytorch官方文档：[PyTorch documentation](https://pytorch.org/docs/stable/index.html)。

## 参考文档

[yunjey/pytorch-tutorial: PyTorch Tutorial for Deep Learning Researchers (github.com)](https://github.com/yunjey/pytorch-tutorial)

[深度学习框架PyTorch常用代码段 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/104019160)

## 基础

**以下所有模块都包含可执行代码。tensor的一些基本操作都包含在应用中，都有相应的注释。**

### torch.tensor与numpy.array、python list之间的相互转化

```python
import torch   # 记得安装pytorch，pip install pytorch / conda install pytorch
import numpy as np

ts = torch.tensor([1,2,3])
arr = np.array([1,2,3])
lst = [1,2,3]

# tensor->list
ts.tolist()
# array->list
arr.tolist()
# list->tensor
torch.tensor(lst)
# array->tensor
torch.from_numpy(arr)
# tensor->array
ts.numpy()
"""
注:tensor通常会被我们放到GPU上进行高效运算，在转换时通常先转到cpu上，同时转成array之后希望做的操作通常是非模型操作，所以使用detach从计算图中拿出
使用：ts.cpu().detach().numpy()
"""
#list->array
np.array(lst)
```

### 模型的输入构造pipeline

Pytorch中，模型输入通常为mini-batch，torch.utils.data.DataLoader可以简单完成这个功能，构造DataLoader对象需要输入继承torch.utils.data.Dataset的dataset对象。构造模型输入的pipeline如下。

```python
# 此部分为模板
# 构造自己的dataset类
from torch.utils.data import Dataset,DataLoader
class MyDataset(Dataset): # 需要继承Dataset
    """
    以下三个函数为自己构造Dataset类时必须定义的三个函数
    """
    
    def __init__(self,mydata):
        self.mydata = mydata  # 将数据集保存到类的对象中
    def __getitem__(self,index):
        """
        输入一个索引index，能获取到这个数据集中第index条数据，若是训练集，通常也返回对应的label
        """
        # 可添加一些处理过程，如tokenizer或其他需要在输入模型前需要做的操作
        # process(self.mydata)
        return self.mydata[index],label[index]
    def __len__(self):   #返回数据集总大小，DataLoader读取长度与其保持一致
        return len(self.mydata)
    
# 接下来对于Dataloader的构建便容易些

# 首先构建MyDataset对象
myDataset = MyDataset(mydata)
# BATCH_SIZE是超参数，自己设定，表示一批数据有多少条数据，shuffle表示是否打乱数据，训练时通常需要shuffle，测试时可将shuffle设置为False，不会影响结果
myDataloader = DataLoader(myDataset,batch_size = BATCH_SIZE, shuffle = True)

# 构建好后，可以使用for循环读取
"""
for data,label in myDataloader:
	output = model(data)
	loss = loss_fn(output,label)
	....
	....
"""
```

##### 可执行测试代码

```python
import torch
from torch.utils.data import Dataset,DataLoader
import random
"""
假设我现在有100条数据，每条数据是一个8维向量，表示输入特征，根据这些特征来判断label
假设我现在的任务是根据一个8维向量来预测输出是0还是1
torch.randn（[shape]）用来生成随机数字的tensor，这些随机数字满足标准正态分布N（0, 1）。
"""
myData_feature = torch.randn([100,8])
myData_label = torch.tensor([random.randint(0,1) for _ in range(100)],dtype = torch.int32)
class MyDataset(Dataset): # 需要继承Dataset
    """
    以下三个函数为自己构造Dataset类时必须定义的三个函数
    """
    def __init__(self,feature,label):
        self.myData_feature = feature
        self.myData_label = label
    def __getitem__(self,index):
        """
        输入一个索引index，能获取到这个数据集中第index条数据，若是训练集，通常也返回对应的label
        """
        return self.myData_feature[index],self.myData_label[index]
    def __len__(self):   #返回数据集总大小，DataLoader读取长度与其保持一致
        return len(self.myData_label)   # 通常，我们的数据集是一条数据对应一个label，因此数据集的大小和label的大小一样
    
myDataset = MyDataset(myData_feature,myData_label)
print("第一条数据的样子：{}".format(myDataset[0]))  # 自动调用__getitem__函数
print("我的数据集长度为：{}".format(len(myDataset)))  # 自动调用__len__函数

# 构建Dataloader
myDataloader = DataLoader(myDataset,batch_size = 4,shuffle = True)   # batch_size为批大小，就是我们通常说的minibatch，梯度下降法更新参数时，
                                                                      #  用的梯度是一批数据的平均梯度
for batch in myDataloader:
    print("一批数据：{}".format(batch))
    print("一批数据的特征形状：{}".format(batch[0].shape))
    print("一批数据的标签形状：{}".format(batch[1].shape))
    break
```



### 模型的构建pipeline

pytorch构建模型，需要继承torch.nn.Module类，从而可以在使用model(data)可以自动执行forward计算过程，因此构建的model通常是包含forward计算过程。

```python
import torch.nn as nn

class myModel(nn.Module):
    def __init__(self,para):  #...表示其他需要初始化的内容，主要为超参数，如需要Linear层，可能需要input_dim,output_dim
        super(myModel,self).__init__()   # 继承nn.Module并初始化
        self.para = para
        self.layer = nn.LayerName(para_needed)
    def forward(self,X):
        output = self.layer(X)
        
        return output  #也可以把loss放在模型中算
```

##### 可执行测试代码

```python
import torch.nn as nn

# 构建一个模型，将“模型的输入构造pipeline”这部分的可执行代码生成的数据作为训练数据，由于是二分类任务，采用逻辑回归即可
class myModel(nn.Module):
    def __init__(self,input_dim,output_dim):  
        super(myModel,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)
        self.acti = nn.Sigmoid()
    def forward(self,X):
        output = self.acti(self.linear(X))
        return output
model = myModel(8,1)  # 目的是根据8维的特征，生成一维的标签
# model(myData_feature[0])
print("第一条数据的特征输出模型，运算结果为：{}".format(model(myData_feature[0])))
for batch in myDataloader:
    print("第一批数据的特征输出模型，运算结果为：{}".format(model(batch[0])))   # batch实际上是一个tuple，第一个元素是feature，第二个元素是label，
                                                                                # 这与我们定义getitem函数的返回值有关
    break
```

### 模型训练pipeline

模型训练需要选用合适的损失函数，损失函数就是我们优化的目标，以求模型的输出和我们数据的label差距比较小。还需要一个优化器，优化器主要负责更新模型参数，因此回顾梯度下降算法，需要给优化器模型的参数以及学习率。

```python
import torch.optim as optim  # pytorch的优化器模块
import torch.nn as nn
"""
设置一系列超参，比如训练的epoch数（num_epochs），学习率(lr)
"""
loss_fn = nn.LossName  # 定义损失函数，即训练的优化目标，loss_fn == loss function
optimizer = optim.OptimizerName  # 定义优化器，即更新模型参数需要的东西

# 训练过程
for epoch in range(num_epochs):
    
    # 定义一些需要统计的量，比如每个epoch的loss，方便跟踪训练过程；每个step的loss都存起来，方便画出训练过程的图
    epoch_loss = 0.0
    
    for batch in myDataloader:  # 按批次读取训练数据
        model.train() # 将模型设置为训练模式
        
        output = model(batch[0])  # 首先进行forward过程，算出当前模型的输出
        loss = loss_fn(output,batch[1])  #利用当前模型的输出和数据的label计算损失
        epoch_loss+=loss.item()  # 将损失加到想看的变量上
        loss.backward()   # 进行backward计算，计算梯度
        optimizer.step() # 利用计算出的梯度更新模型参数
        optimizer.zero_grad()  # pytorch的optimizer梯度会累积，所以每个step需要将梯度清空，不影响下一次的操作
        
        # 后面可以接上一些valid过程，验证模型的效果
        """
        model.eval() # 设置为评估模式
        with torch.no_grad():   # 验证的过程不需要计算梯度
        	验证过程代码
        """
    print("Epoch:{}, loss:{}".format(epoch+1,epoch_loss))  # 一定程度输出一次训练相关信息，也可以每10个或每100个step输出一次，根据训练速度来
```

#### 可执行代码

```python
import torch.optim as optim  # pytorch的优化器模块
import torch.nn as nn

loss_fn = nn.MSELoss()  # 使用 L2loss
optimizer = optim.SGD(model.parameters(),lr = 1e-2)   # 定义优化器，传入模型参数，学习率lr，常用优化器为optim.SGD, optim.Adam
num_epochs = 1000

for epoch in range(num_epochs): # 一个epoch表示一轮训练，完整的使用一遍参数，几个epoch就扫几次数据
    epoch_loss = 0.0  # 统计一个epoch总的loss，输出方便看模型效果，也可以统计其他信息，如每个step的loss放在一个list中，方便画图看训练过程
    
    for batch in myDataloader:
        model.train()
        
        output = model(batch[0])  
        loss = loss_fn(output,batch[1])  
        epoch_loss+=loss.item() 
        loss.backward()   
        optimizer.step() 
        optimizer.zero_grad() 
        
        # 后面可以接上一些valid过程，验证模型的效果
        """
        model.eval() # 设置为评估模式
        with torch.no_grad():   # 验证的过程不需要计算梯度
        	验证过程代码
        """
    print("Epoch:{}, loss:{}".format(epoch+1,epoch_loss))  # 训练到一定程度（此处代码是一个epoch）输出一次训练相关信息，也可以每10个或每100个step输出一次，根据训练速度来
    
    
"""
由于数据都是随机生成的，模型即使训练后可能输出也在0.5左右
"""
```

