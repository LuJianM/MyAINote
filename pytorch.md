一个分为如下几步















查看pytorch版本

```python
import torch
print(torch.__version__)	# torch版本
print(torch.cuda.is_available())	# GPU是否可用
print(torch.cuda.device_count())	# GPU个数
print(torch.cuda.get_device_name())	# GPU名称

'''
1.5.0
True
1
GeForce GTX 1050 Ti
'''
```



生成一个5*3的矩阵

```python
a = torch.empty(5,3)
print(a)
'''
tensor([[-1.0954e+00,  8.1415e-43, -1.0954e+00],
        [ 8.1415e-43, -1.0954e+00,  8.1415e-43],
        [-1.0954e+00,  8.1415e-43, -1.0954e+00],
        [ 8.1415e-43, -1.0954e+00,  8.1415e-43],
        [-1.0954e+00,  8.1415e-43, -1.0954e+00]])
'''
```







将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 **x**) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 **C**，如下图。单词向量矩阵用 **X(n×d)**表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。



### epoch与batch

一个epoch意味着训练集中每一个样本都参与训练了一次。



对于batch

比如想在有1000个数据，但是每次训练只扔进去50个，即batch size 为50。那么我的batch就应该为200。