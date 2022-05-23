# Pytorch基础教程与常用操作

本教程基于本人常用的pytorch操作进行整理，用于帮助快速入门与某些torch技巧的查阅，仅限参考。

深入学习请多花时间阅读pytorch官方文档：[PyTorch documentation](https://pytorch.org/docs/stable/index.html)。

## 参考文档

[yunjey/pytorch-tutorial: PyTorch Tutorial for Deep Learning Researchers (github.com)](https://github.com/yunjey/pytorch-tutorial)

[[深度学习框架\]PyTorch常用代码段 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/104019160)

## 基础

#### 以下所有模块都假设已import相应的模块

### torch.tensor与numpy.array、python list之间的相互转化

```python
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
        return self.mydata,label
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

### 模型的构建pipeline

```python

```

