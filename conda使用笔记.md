# conda使用笔记

```
conda info -e：查看有哪些环境
conda list：查看当前环境有哪些包
conda env remove -n mypytorch ： 删除环境 其中mypytorch是环境名称
conda activate env_name: 激活环境
```



创建新环境：`conda create -n python3.6 python=3.6`

进入某个环境：`conda activate python3.6`

安装某个包：`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

验证torch是否安装成功

```python
import torch 
print(torch.cuda.is_available())
```

安装`sklearn`：`conda install scikit-learn`

安装`tqdm`：`conda install tqdm`

安装`tensorboardX`: `conda install -c conda-forge tensorboardx `





### 安装各种包

#### 安装cv2

从网上下载[http://pypi.doubanio.com/simple/]

```
opencv_python-4.0.1.24-cp36-cp36m-win_amd64
```

包名-版本号-适合Python版本(36表示Python3.6)-系统（win_amd64表示windows64位）



安装 

```bash
pip install F:\AI\opencv_python-4.0.1.24-cp36-cp36m-win_amd64.whl
```





#### 安装cython





#### 安装tensorflow

安装指定版本 `conda install tensorflow==1.12.0`









#### 安装yaml

```bash
conda install pyyaml
```



#### 安装PIL

```bash
conda install Pillow
```



#### 安装lmdb

```bash
conda install python-lmdb
```



#### 安装python-Levenshtein 

```bash
conda install python-Levenshtein 
```



```bash
conda install easydict 
```





安装selenium

``



#### 安装`pytesseract`

```

```





安装`tensorboard`

```bash
(python3.6) C:\MyPython\Chinese-Text-Classification-Pytorch-master\THUCNews\log\Transformer>conda install tensorboard
Collecting package metadata (current_repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.8.3
  latest version: 4.9.2

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: F:\AI\anaconda3\envs\python3.6

  added / updated specs:
    - tensorboard


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    absl-py-0.11.0             |   py36haa95532_0         192 KB  defaults
    ca-certificates-2020.10.14 |                0         122 KB  defaults
    grpcio-1.31.0              |   py36he7da953_0         1.5 MB  defaults
    markdown-2.6.9             |           py36_0         100 KB  https://repo.continuum.io/pkgs/free
    openssl-1.1.1h             |       he774522_0         4.8 MB  defaults
    tensorboard-1.14.0         |   py36he3c9ec2_0         3.1 MB  defaults
    werkzeug-0.12.2            |           py36_0         435 KB  https://repo.continuum.io/pkgs/free
    ------------------------------------------------------------
                                           Total:        10.3 MB

The following NEW packages will be INSTALLED:

  absl-py            pkgs/main/win-64::absl-py-0.11.0-py36haa95532_0
  ca-certificates    pkgs/main/win-64::ca-certificates-2020.10.14-0
  grpcio             pkgs/main/win-64::grpcio-1.31.0-py36he7da953_0
  markdown           pkgs/free/win-64::markdown-2.6.9-py36_0
  openssl            pkgs/main/win-64::openssl-1.1.1h-he774522_0
  tensorboard        pkgs/main/win-64::tensorboard-1.14.0-py36he3c9ec2_0
  werkzeug           pkgs/free/win-64::werkzeug-0.12.2-py36_0

The following packages will be SUPERSEDED by a higher-priority channel:

  certifi            conda-forge::certifi-2020.11.8-py36ha~ --> pkgs/free::certifi-2016.2.28-py36_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
ca-certificates-2020 | 122 KB    | ################################################################################################################################################# | 100%
grpcio-1.31.0        | 1.5 MB    | ################################################################################################################################################# | 100%
tensorboard-1.14.0   | 3.1 MB    | ################################################################################################################################################# | 100%
absl-py-0.11.0       | 192 KB    | ################################################################################################################################################# | 100%
werkzeug-0.12.2      | 435 KB    | ################################################################################################################################################# | 100%
openssl-1.1.1h       | 4.8 MB    | ################################################################################################################################################# | 100%
markdown-2.6.9       | 100 KB    | ################################################################################################################################################# | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done

(python3.6) C:\MyPython\Chinese-Text-Classification-Pytorch-master\THUCNews\log\Transformer>
```



问题：

```
No module named 'overrides'
```

安装：

```



运行结果：

​```python
C:\AI\anaconda3\envs\python3.6\python.exe C:/MyPython/Chinese-Text-Classification-Pytorch-master/run.py --model Transformer
Transformer
<class 'str'>
Vocab size: 4762
180000it [00:02, 72716.31it/s]
10000it [00:00, 49877.27it/s]
10000it [00:00, 74810.25it/s]
Epoch [1/20]
Iter:      0,  Train Loss:   2.3,  Train Acc:  9.38%,  Val Loss:   2.7,  Val Acc: 10.00%,  Time: 0:00:03 *
Iter:    100,  Train Loss:  0.77,  Train Acc: 71.09%,  Val Loss:   0.7,  Val Acc: 78.09%,  Time: 0:00:11 *
Iter:    200,  Train Loss:  0.75,  Train Acc: 76.56%,  Val Loss:  0.56,  Val Acc: 82.90%,  Time: 0:00:20 *
Iter:    300,  Train Loss:  0.48,  Train Acc: 86.72%,  Val Loss:  0.49,  Val Acc: 84.63%,  Time: 0:00:28 *
Iter:    400,  Train Loss:  0.73,  Train Acc: 78.91%,  Val Loss:  0.47,  Val Acc: 85.52%,  Time: 0:00:36 *
Iter:    500,  Train Loss:  0.41,  Train Acc: 87.50%,  Val Loss:  0.43,  Val Acc: 86.44%,  Time: 0:00:44 *
Iter:    600,  Train Loss:  0.52,  Train Acc: 83.59%,  Val Loss:  0.43,  Val Acc: 86.79%,  Time: 0:00:52 *
Iter:    700,  Train Loss:  0.45,  Train Acc: 85.16%,  Val Loss:   0.4,  Val Acc: 87.41%,  Time: 0:01:00 *
Iter:    800,  Train Loss:   0.5,  Train Acc: 85.94%,  Val Loss:  0.39,  Val Acc: 87.83%,  Time: 0:01:08 *
Iter:    900,  Train Loss:   0.4,  Train Acc: 88.28%,  Val Loss:  0.38,  Val Acc: 88.22%,  Time: 0:01:17 *
Iter:   1000,  Train Loss:  0.29,  Train Acc: 89.06%,  Val Loss:  0.39,  Val Acc: 87.83%,  Time: 0:01:25 
Iter:   1100,  Train Loss:  0.37,  Train Acc: 89.84%,  Val Loss:  0.38,  Val Acc: 88.29%,  Time: 0:01:33 *
Iter:   1200,  Train Loss:  0.31,  Train Acc: 87.50%,  Val Loss:  0.37,  Val Acc: 88.67%,  Time: 0:01:41 *
Iter:   1300,  Train Loss:  0.44,  Train Acc: 84.38%,  Val Loss:  0.36,  Val Acc: 88.73%,  Time: 0:01:49 *
Iter:   1400,  Train Loss:  0.47,  Train Acc: 85.16%,  Val Loss:  0.36,  Val Acc: 88.90%,  Time: 0:01:57 *
Epoch [2/20]
Iter:   1500,  Train Loss:  0.51,  Train Acc: 85.94%,  Val Loss:  0.35,  Val Acc: 88.88%,  Time: 0:02:06 *
Iter:   1600,  Train Loss:  0.26,  Train Acc: 90.62%,  Val Loss:  0.35,  Val Acc: 89.08%,  Time: 0:02:14 *
Iter:   1700,  Train Loss:  0.38,  Train Acc: 85.94%,  Val Loss:  0.34,  Val Acc: 89.26%,  Time: 0:02:22 *
Iter:   1800,  Train Loss:  0.39,  Train Acc: 89.06%,  Val Loss:  0.36,  Val Acc: 88.80%,  Time: 0:02:30 
Iter:   1900,  Train Loss:  0.36,  Train Acc: 89.06%,  Val Loss:  0.34,  Val Acc: 89.59%,  Time: 0:02:38 *
Iter:   2000,  Train Loss:  0.41,  Train Acc: 87.50%,  Val Loss:  0.34,  Val Acc: 89.46%,  Time: 0:02:47 
Iter:   2100,  Train Loss:  0.33,  Train Acc: 90.62%,  Val Loss:  0.34,  Val Acc: 89.58%,  Time: 0:02:55 
Iter:   2200,  Train Loss:  0.33,  Train Acc: 86.72%,  Val Loss:  0.34,  Val Acc: 89.43%,  Time: 0:03:03 *
Iter:   2300,  Train Loss:  0.32,  Train Acc: 91.41%,  Val Loss:  0.35,  Val Acc: 89.13%,  Time: 0:03:11 
Iter:   2400,  Train Loss:   0.3,  Train Acc: 90.62%,  Val Loss:  0.34,  Val Acc: 89.67%,  Time: 0:03:20 
Iter:   2500,  Train Loss:  0.18,  Train Acc: 95.31%,  Val Loss:  0.33,  Val Acc: 89.76%,  Time: 0:03:28 *
Iter:   2600,  Train Loss:  0.41,  Train Acc: 85.94%,  Val Loss:  0.33,  Val Acc: 89.81%,  Time: 0:03:36 
Iter:   2700,  Train Loss:  0.28,  Train Acc: 89.06%,  Val Loss:  0.33,  Val Acc: 89.61%,  Time: 0:03:44 
Iter:   2800,  Train Loss:  0.44,  Train Acc: 85.94%,  Val Loss:  0.34,  Val Acc: 89.73%,  Time: 0:03:52 
Epoch [3/20]
Iter:   2900,  Train Loss:  0.37,  Train Acc: 89.06%,  Val Loss:  0.33,  Val Acc: 89.86%,  Time: 0:04:01 
Iter:   3000,  Train Loss:  0.21,  Train Acc: 92.97%,  Val Loss:  0.33,  Val Acc: 89.93%,  Time: 0:04:09 
Iter:   3100,  Train Loss:  0.32,  Train Acc: 90.62%,  Val Loss:  0.33,  Val Acc: 89.94%,  Time: 0:04:17 *
Iter:   3200,  Train Loss:  0.41,  Train Acc: 88.28%,  Val Loss:  0.33,  Val Acc: 89.93%,  Time: 0:04:26 
Iter:   3300,  Train Loss:  0.27,  Train Acc: 90.62%,  Val Loss:  0.32,  Val Acc: 90.23%,  Time: 0:04:34 *
Iter:   3400,  Train Loss:  0.36,  Train Acc: 87.50%,  Val Loss:  0.33,  Val Acc: 89.96%,  Time: 0:04:42 
Iter:   3500,  Train Loss:  0.25,  Train Acc: 92.19%,  Val Loss:  0.32,  Val Acc: 90.33%,  Time: 0:04:50 
Iter:   3600,  Train Loss:  0.21,  Train Acc: 94.53%,  Val Loss:  0.32,  Val Acc: 90.06%,  Time: 0:04:59 
Iter:   3700,  Train Loss:  0.32,  Train Acc: 87.50%,  Val Loss:  0.32,  Val Acc: 90.32%,  Time: 0:05:07 *
Iter:   3800,  Train Loss:  0.42,  Train Acc: 82.81%,  Val Loss:  0.33,  Val Acc: 89.77%,  Time: 0:05:16 
Iter:   3900,  Train Loss:  0.33,  Train Acc: 88.28%,  Val Loss:  0.32,  Val Acc: 90.20%,  Time: 0:05:24 
Iter:   4000,  Train Loss:  0.26,  Train Acc: 91.41%,  Val Loss:  0.33,  Val Acc: 90.18%,  Time: 0:05:32 
Iter:   4100,  Train Loss:  0.27,  Train Acc: 90.62%,  Val Loss:  0.32,  Val Acc: 90.39%,  Time: 0:05:41 
Iter:   4200,  Train Loss:  0.27,  Train Acc: 92.19%,  Val Loss:  0.33,  Val Acc: 90.00%,  Time: 0:05:49 
Epoch [4/20]
Iter:   4300,  Train Loss:  0.22,  Train Acc: 91.41%,  Val Loss:  0.32,  Val Acc: 90.12%,  Time: 0:05:57 
Iter:   4400,  Train Loss:  0.22,  Train Acc: 93.75%,  Val Loss:  0.32,  Val Acc: 90.36%,  Time: 0:06:06 
Iter:   4500,  Train Loss:   0.3,  Train Acc: 89.84%,  Val Loss:  0.32,  Val Acc: 90.48%,  Time: 0:06:14 
Iter:   4600,  Train Loss:  0.27,  Train Acc: 90.62%,  Val Loss:  0.32,  Val Acc: 90.21%,  Time: 0:06:22 
Iter:   4700,  Train Loss:  0.35,  Train Acc: 90.62%,  Val Loss:  0.32,  Val Acc: 90.32%,  Time: 0:06:31 
No optimization for a long time, auto-stopping...
Test Loss:   0.3,  Test Acc: 90.94%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      	header    0.9260    0.8880    0.9066      1000
       realty     0.9010    0.9460    0.9229      1000
       stocks     0.8519    0.8570    0.8544      1000
    education     0.9457    0.9580    0.9518      1000
      science     0.8722    0.8670    0.8696      1000
      society     0.8937    0.9250    0.9091      1000
     politics     0.9051    0.8770    0.8908      1000
       sports     0.9342    0.9650    0.9493      1000
         game     0.9335    0.8990    0.9159      1000
entertainment     0.9325    0.9120    0.9221      1000

     accuracy                         0.9094     10000
    macro avg     0.9096    0.9094    0.9093     10000
 weighted avg     0.9096    0.9094    0.9093     10000

Confusion Matrix...
[[888  19  54   5   9   8   7   5   2   3]
 [  9 946  14   2   1  13   3   4   1   7]
 [ 42  26 857   4  27   2  35   3   3   1]
 [  1   3   2 958   2  15   5   4   0  10]
 [  2  12  31   7 867  18  16   5  31  11]
 [  3  18   2  16   7 925  15   2   3   9]
 [  9  11  30  12  15  34 877   4   2   6]
 [  1   2   4   1   5   5   3 965   2  12]
 [  0   4   9   3  53   8   2  15 899   7]
 [  4   9   3   5   8   7   6  26  20 912]]
Time usage: 0:00:01

Process finished with exit code 0
```





## argparse命令行解释器

```python
# 1. 导包
import argparse

# 2. 生成一个命令行参数对象，argparse.ArgumentParser
parser = argparse.ArgumentParser(description='Chinese Text Classification')
#
"""
3. 往参数对象中添加三个参数，分别为：
    model（指定训练模型）
    embedding（编码方式，预训练与随机）
    word（bool值，True为word，False为char）
"""
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
"""
3. 转化为名称空间
"""
args = parser.parse_args()

print("parser的类型为： ", type(parser))	# argparse.Namespace
print("args的类型为：", type(args))	# argparse.Namespace

# 4.取出参数
print(args.model)
```







## Transformer配置参数

模型名称

训练集路径

验证集路径

测试集路径

类别名单路径

词表路径

模型训练结果路径

日志路径

预训练词向量





## pycharm

### 关闭代码风格检查

`setting–>Inspections–>Python–>PEP8`









# pytorch

## 1. torch.Tensor

得到tensor的信息：

`tensor.shape`





### 模型的保存与读取

常用的三个函数

- `torch.save()`: 保存一个序列化的对象到磁盘，使用的是`Python`的`pickle`库来实现的
- `torch.load()`: 解序列化一个`pickled`对象并加载到内存当中
- `torch.nn.Module.load_state_dict()`: 加载一个解序列化的`state_dict`对象

#### 1. state_dict

在`PyTorch`中所有可学习的参数保存在`model.parameters()`中。`state_dict`是一个`Python`字典。保存了各层与其参数张量之间的映射。`torch.optim`对象也有一个`state_dict`，它包含了`optimizer`的`state`，以及一些超参数。



#### 2. 模型的保存

**保存**

```python
torch.save(model.state_dict(), path)
```

**加载**

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(path))
model.eval()  # 当用于inference时不要忘记添加
```







#### 3. 模型的保存与读取（保存模型+参数）







## TensorboardX

### 创建SummaryWriter 

```python
from tensorboardX import SummaryWriter

# Creates writer1 object.
# The log will be saved in 'runs/exp'
writer1 = SummaryWriter('runs/exp')

# Creates writer2 object with auto generated file name
# The log directory will be something like 'runs/Aug20-17-20-33'
writer2 = SummaryWriter()

# Creates writer3 object with auto generated file name, the comment will be appended to the filename.
writer3 = SummaryWriter(comment='resnet')



# 2使用 add_scalar 方法来记录数字常量。


```

以上展示了三种初始化 SummaryWriter 的方法：

1. 提供一个路径，将使用该路径来保存日志
2. 无参数，默认将使用 `runs/日期时间` 路径来保存日志
3. 提供一个 comment 参数，将使用 `runs/日期时间-comment` 路径来保存日志



**参数**

- **tag** (string): 数据名称，不同名称的数据使用不同曲线展示
- **scalar_value** (float): 数字常量值
- **global_step** (int, optional): 训练的 step
- **walltime** (float, optional): 记录发生的时间，默认为 `time.time()`



## 使用tensorboard将保存的日志文件进行可视化

**安装tensorboard**

1. tensorboard与TensorboardX是两个东西

没安装tensorboard，会报如下错误

```bash
(python3.6) C:\MyPython\Chinese-Text-Classification-Pytorch-master\THUCNews\log\Transformer>tensorboard --logdir=11-21_11
'tensorboard' 不是内部或外部命令，也不是可运行的程序
或批处理文件。

```

去虚拟环境中的`Scripts`没找到可执行的文件`tensorboard.exe`和`tensorboard-script.py`。

使用conda安装tensorboard

`conda install tensorboard`



**启动tensorboard**

启动命令

`tensorboard --logdir="日志文件所在文件夹的绝对路径" --host=127.0.0.1`

```bash
tensorboard --logdir="C:\MyPython\Chinese-Text-Classification-Pytorch-master\THUCNews\log\Transformer\11-21_11.14" --host=127.0.0.1
```

需要注意的：

1. 最好指定`ip`地址，为127.0.0.1。不然会以默认电脑名称为域名，有可能变得无法访问
2. 参数`logdir`为文件所在文件夹的绝对路径
3. 文件夹绝对路径一定要加上双引号





```python
tensorboard --logdir="C:\MyPython\Chinese-Text-Classification-Pytorch-master\THUCNews\log\TextCNN\11-19_11.57" --host=127.0.0.1
```







# Transformer

## 位置编码

### 为什么要位置编码

对于一个表格中的单元格来说，单元格在表格中的位置是非常重要的，它不仅仅是表格的组成成分的一种，更是表达语义的重要概念。一个单元格在表格中的位置或者排列顺序不同，它表示的意义可能不同。























# 实验

## 实验结果

#### 准确率 0.8

```bash
F:\AI\anaconda3\envs\python3.6\python.exe C:/MyPython/Chinese-Text-Classification-Pytorch-master/run.py --model Transformer
Loading data...
Vocab size: 4762
180000it [00:03, 55567.02it/s]
10000it [00:00, 47971.20it/s]
10000it [00:00, 61136.47it/s]
批次大小（一次喂入的数据量）： 128
使用的设备： cuda
<class 'utils.DatasetIterater'>
Time usage: 0:00:04
model的类型： <class 'models.Transformer.Model'>
Epoch [1/20]
Iter:      0,  Train Loss:   2.5,  Train Acc:  7.81%,  Val Loss:   2.5,  Val Acc: 10.69%,  Time: 0:00:02 *
Iter:    100,  Train Loss:   1.8,  Train Acc: 39.06%,  Val Loss:   2.0,  Val Acc: 30.77%,  Time: 0:00:12 *
Iter:    200,  Train Loss:   1.5,  Train Acc: 49.22%,  Val Loss:   1.7,  Val Acc: 43.78%,  Time: 0:00:22 *
Iter:    300,  Train Loss:   1.2,  Train Acc: 62.50%,  Val Loss:   1.7,  Val Acc: 49.73%,  Time: 0:00:32 *
Iter:    400,  Train Loss:   1.1,  Train Acc: 64.06%,  Val Loss:   1.6,  Val Acc: 52.54%,  Time: 0:00:42 *
Iter:    500,  Train Loss:  0.86,  Train Acc: 75.00%,  Val Loss:   1.5,  Val Acc: 57.18%,  Time: 0:00:52 *
Iter:    600,  Train Loss:  0.96,  Train Acc: 67.19%,  Val Loss:   1.4,  Val Acc: 59.46%,  Time: 0:01:01 *
Iter:    700,  Train Loss:   1.1,  Train Acc: 63.28%,  Val Loss:   1.4,  Val Acc: 61.33%,  Time: 0:01:11 *
Iter:    800,  Train Loss:  0.79,  Train Acc: 80.47%,  Val Loss:   1.4,  Val Acc: 61.73%,  Time: 0:01:21 
Iter:    900,  Train Loss:  0.82,  Train Acc: 71.09%,  Val Loss:   1.3,  Val Acc: 64.48%,  Time: 0:01:31 *
Iter:   1000,  Train Loss:  0.71,  Train Acc: 79.69%,  Val Loss:   1.5,  Val Acc: 63.04%,  Time: 0:01:41 
Iter:   1100,  Train Loss:  0.71,  Train Acc: 77.34%,  Val Loss:   1.3,  Val Acc: 65.88%,  Time: 0:01:52 *
Iter:   1200,  Train Loss:  0.67,  Train Acc: 78.12%,  Val Loss:   1.3,  Val Acc: 65.88%,  Time: 0:02:02 
Iter:   1300,  Train Loss:  0.78,  Train Acc: 76.56%,  Val Loss:   1.3,  Val Acc: 65.90%,  Time: 0:02:12 *
Iter:   1400,  Train Loss:   0.9,  Train Acc: 71.09%,  Val Loss:   1.3,  Val Acc: 65.91%,  Time: 0:02:22 
Epoch [2/20]
Iter:   1500,  Train Loss:  0.79,  Train Acc: 75.78%,  Val Loss:   1.3,  Val Acc: 65.19%,  Time: 0:02:33 
Iter:   1600,  Train Loss:  0.69,  Train Acc: 73.44%,  Val Loss:   1.3,  Val Acc: 65.92%,  Time: 0:02:43 
Iter:   1700,  Train Loss:  0.81,  Train Acc: 76.56%,  Val Loss:   1.0,  Val Acc: 71.77%,  Time: 0:02:53 *
Iter:   1800,  Train Loss:  0.54,  Train Acc: 83.59%,  Val Loss:   1.4,  Val Acc: 64.28%,  Time: 0:03:04 
Iter:   1900,  Train Loss:  0.66,  Train Acc: 75.78%,  Val Loss:   1.3,  Val Acc: 66.47%,  Time: 0:03:14 
Iter:   2000,  Train Loss:  0.69,  Train Acc: 80.47%,  Val Loss:   1.1,  Val Acc: 69.23%,  Time: 0:03:24 
Iter:   2100,  Train Loss:  0.72,  Train Acc: 76.56%,  Val Loss:   1.1,  Val Acc: 71.14%,  Time: 0:03:35 
Iter:   2200,  Train Loss:  0.62,  Train Acc: 76.56%,  Val Loss:   1.2,  Val Acc: 68.76%,  Time: 0:03:45 
Iter:   2300,  Train Loss:  0.64,  Train Acc: 78.12%,  Val Loss:   1.3,  Val Acc: 67.28%,  Time: 0:03:55 
Iter:   2400,  Train Loss:  0.69,  Train Acc: 78.12%,  Val Loss:   1.2,  Val Acc: 69.64%,  Time: 0:04:06 
Iter:   2500,  Train Loss:  0.64,  Train Acc: 80.47%,  Val Loss:   1.4,  Val Acc: 65.81%,  Time: 0:04:16 
Iter:   2600,  Train Loss:  0.76,  Train Acc: 77.34%,  Val Loss:  0.98,  Val Acc: 73.52%,  Time: 0:04:26 *
Iter:   2700,  Train Loss:  0.56,  Train Acc: 82.03%,  Val Loss:   1.2,  Val Acc: 70.15%,  Time: 0:04:37 
Iter:   2800,  Train Loss:  0.71,  Train Acc: 75.78%,  Val Loss:  0.87,  Val Acc: 75.51%,  Time: 0:04:47 *
Epoch [3/20]
Iter:   2900,  Train Loss:  0.52,  Train Acc: 82.81%,  Val Loss:   1.0,  Val Acc: 72.62%,  Time: 0:04:58 
Iter:   3000,  Train Loss:  0.71,  Train Acc: 77.34%,  Val Loss:   1.0,  Val Acc: 72.70%,  Time: 0:05:08 
Iter:   3100,  Train Loss:  0.59,  Train Acc: 81.25%,  Val Loss:   1.2,  Val Acc: 70.60%,  Time: 0:05:19 
Iter:   3200,  Train Loss:  0.74,  Train Acc: 83.59%,  Val Loss:  0.95,  Val Acc: 74.03%,  Time: 0:05:29 
Iter:   3300,  Train Loss:  0.58,  Train Acc: 81.25%,  Val Loss:   1.1,  Val Acc: 71.93%,  Time: 0:05:39 
Iter:   3400,  Train Loss:  0.79,  Train Acc: 74.22%,  Val Loss:   1.0,  Val Acc: 73.43%,  Time: 0:05:50 
Iter:   3500,  Train Loss:  0.51,  Train Acc: 79.69%,  Val Loss:   1.1,  Val Acc: 71.52%,  Time: 0:06:00 
Iter:   3600,  Train Loss:  0.35,  Train Acc: 87.50%,  Val Loss:  0.91,  Val Acc: 75.15%,  Time: 0:06:10 
Iter:   3700,  Train Loss:  0.65,  Train Acc: 72.66%,  Val Loss:  0.98,  Val Acc: 73.86%,  Time: 0:06:21 
Iter:   3800,  Train Loss:  0.54,  Train Acc: 81.25%,  Val Loss:  0.86,  Val Acc: 76.22%,  Time: 0:06:31 *
Iter:   3900,  Train Loss:  0.71,  Train Acc: 78.12%,  Val Loss:  0.88,  Val Acc: 75.74%,  Time: 0:06:42 
Iter:   4000,  Train Loss:  0.51,  Train Acc: 82.81%,  Val Loss:   1.1,  Val Acc: 71.96%,  Time: 0:06:52 
Iter:   4100,  Train Loss:  0.65,  Train Acc: 78.91%,  Val Loss:  0.99,  Val Acc: 74.96%,  Time: 0:07:03 
Iter:   4200,  Train Loss:  0.75,  Train Acc: 79.69%,  Val Loss:  0.96,  Val Acc: 74.21%,  Time: 0:07:13 
Epoch [4/20]
Iter:   4300,  Train Loss:   0.5,  Train Acc: 84.38%,  Val Loss:  0.99,  Val Acc: 73.94%,  Time: 0:07:24 
Iter:   4400,  Train Loss:  0.38,  Train Acc: 87.50%,  Val Loss:  0.84,  Val Acc: 77.33%,  Time: 0:07:34 *
Iter:   4500,  Train Loss:  0.56,  Train Acc: 82.03%,  Val Loss:   1.0,  Val Acc: 74.92%,  Time: 0:07:44 
Iter:   4600,  Train Loss:  0.57,  Train Acc: 78.12%,  Val Loss:   1.1,  Val Acc: 73.28%,  Time: 0:07:55 
Iter:   4700,  Train Loss:  0.72,  Train Acc: 75.78%,  Val Loss:  0.99,  Val Acc: 74.65%,  Time: 0:08:05 
Iter:   4800,  Train Loss:  0.49,  Train Acc: 84.38%,  Val Loss:  0.84,  Val Acc: 77.64%,  Time: 0:08:16 
Iter:   4900,  Train Loss:  0.49,  Train Acc: 83.59%,  Val Loss:  0.93,  Val Acc: 75.61%,  Time: 0:08:26 
Iter:   5000,  Train Loss:  0.46,  Train Acc: 85.16%,  Val Loss:   1.1,  Val Acc: 72.81%,  Time: 0:08:37 
Iter:   5100,  Train Loss:  0.59,  Train Acc: 82.03%,  Val Loss:   1.0,  Val Acc: 74.18%,  Time: 0:08:47 
Iter:   5200,  Train Loss:  0.71,  Train Acc: 75.78%,  Val Loss:   1.0,  Val Acc: 75.25%,  Time: 0:08:58 
Iter:   5300,  Train Loss:  0.35,  Train Acc: 87.50%,  Val Loss:   1.2,  Val Acc: 71.98%,  Time: 0:09:08 
Iter:   5400,  Train Loss:  0.77,  Train Acc: 80.47%,  Val Loss:  0.78,  Val Acc: 79.28%,  Time: 0:09:19 *
Iter:   5500,  Train Loss:   0.5,  Train Acc: 82.81%,  Val Loss:  0.91,  Val Acc: 76.98%,  Time: 0:09:30 
Iter:   5600,  Train Loss:  0.43,  Train Acc: 89.06%,  Val Loss:  0.82,  Val Acc: 78.17%,  Time: 0:09:40 
Epoch [5/20]
Iter:   5700,  Train Loss:   0.6,  Train Acc: 78.91%,  Val Loss:  0.88,  Val Acc: 76.56%,  Time: 0:09:51 
Iter:   5800,  Train Loss:  0.36,  Train Acc: 88.28%,  Val Loss:  0.77,  Val Acc: 79.28%,  Time: 0:10:01 *
Iter:   5900,  Train Loss:  0.41,  Train Acc: 85.94%,  Val Loss:  0.79,  Val Acc: 78.92%,  Time: 0:10:12 
Iter:   6000,  Train Loss:   0.5,  Train Acc: 84.38%,  Val Loss:  0.76,  Val Acc: 79.24%,  Time: 0:10:22 *
Iter:   6100,  Train Loss:  0.52,  Train Acc: 85.16%,  Val Loss:  0.95,  Val Acc: 75.81%,  Time: 0:10:33 
Iter:   6200,  Train Loss:  0.43,  Train Acc: 87.50%,  Val Loss:  0.94,  Val Acc: 75.94%,  Time: 0:10:44 
Iter:   6300,  Train Loss:  0.44,  Train Acc: 84.38%,  Val Loss:  0.81,  Val Acc: 78.38%,  Time: 0:10:54 
Iter:   6400,  Train Loss:  0.32,  Train Acc: 91.41%,  Val Loss:  0.82,  Val Acc: 78.23%,  Time: 0:11:05 
Iter:   6500,  Train Loss:   0.5,  Train Acc: 85.16%,  Val Loss:  0.81,  Val Acc: 78.85%,  Time: 0:11:15 
Iter:   6600,  Train Loss:  0.51,  Train Acc: 82.81%,  Val Loss:   0.8,  Val Acc: 79.02%,  Time: 0:11:26 
Iter:   6700,  Train Loss:  0.46,  Train Acc: 85.16%,  Val Loss:   0.9,  Val Acc: 76.96%,  Time: 0:11:36 
Iter:   6800,  Train Loss:  0.55,  Train Acc: 82.81%,  Val Loss:  0.88,  Val Acc: 78.15%,  Time: 0:11:47 
Iter:   6900,  Train Loss:  0.42,  Train Acc: 82.03%,  Val Loss:  0.81,  Val Acc: 78.88%,  Time: 0:11:57 
Iter:   7000,  Train Loss:  0.52,  Train Acc: 83.59%,  Val Loss:   1.0,  Val Acc: 74.71%,  Time: 0:12:08 
Epoch [6/20]
Iter:   7100,  Train Loss:  0.37,  Train Acc: 88.28%,  Val Loss:   1.0,  Val Acc: 75.33%,  Time: 0:12:19 
Iter:   7200,  Train Loss:  0.52,  Train Acc: 81.25%,  Val Loss:  0.89,  Val Acc: 77.30%,  Time: 0:12:29 
Iter:   7300,  Train Loss:  0.46,  Train Acc: 85.16%,  Val Loss:   1.1,  Val Acc: 74.97%,  Time: 0:12:40 
Iter:   7400,  Train Loss:  0.52,  Train Acc: 79.69%,  Val Loss:  0.94,  Val Acc: 76.80%,  Time: 0:12:50 
Iter:   7500,  Train Loss:  0.47,  Train Acc: 86.72%,  Val Loss:  0.79,  Val Acc: 79.17%,  Time: 0:13:01 
Iter:   7600,  Train Loss:  0.45,  Train Acc: 85.16%,  Val Loss:  0.72,  Val Acc: 80.79%,  Time: 0:13:12 *
Iter:   7700,  Train Loss:  0.58,  Train Acc: 81.25%,  Val Loss:  0.77,  Val Acc: 79.84%,  Time: 0:13:22 
Iter:   7800,  Train Loss:  0.56,  Train Acc: 81.25%,  Val Loss:  0.88,  Val Acc: 78.22%,  Time: 0:13:33 
Iter:   7900,  Train Loss:  0.36,  Train Acc: 87.50%,  Val Loss:  0.88,  Val Acc: 78.02%,  Time: 0:13:43 
Iter:   8000,  Train Loss:  0.43,  Train Acc: 85.16%,  Val Loss:  0.76,  Val Acc: 79.93%,  Time: 0:13:54 
Iter:   8100,  Train Loss:  0.32,  Train Acc: 87.50%,  Val Loss:  0.76,  Val Acc: 79.94%,  Time: 0:14:05 
Iter:   8200,  Train Loss:  0.46,  Train Acc: 87.50%,  Val Loss:  0.64,  Val Acc: 82.61%,  Time: 0:14:15 *
Iter:   8300,  Train Loss:  0.35,  Train Acc: 89.84%,  Val Loss:  0.76,  Val Acc: 80.59%,  Time: 0:14:26 
Iter:   8400,  Train Loss:  0.64,  Train Acc: 75.78%,  Val Loss:  0.79,  Val Acc: 79.71%,  Time: 0:14:37 
Epoch [7/20]
Iter:   8500,  Train Loss:  0.62,  Train Acc: 81.25%,  Val Loss:   1.0,  Val Acc: 75.82%,  Time: 0:14:47 
Iter:   8600,  Train Loss:  0.43,  Train Acc: 85.94%,  Val Loss:  0.85,  Val Acc: 79.26%,  Time: 0:14:58 
Iter:   8700,  Train Loss:  0.34,  Train Acc: 90.62%,  Val Loss:  0.74,  Val Acc: 81.02%,  Time: 0:15:08 
Iter:   8800,  Train Loss:  0.46,  Train Acc: 81.25%,  Val Loss:  0.82,  Val Acc: 79.65%,  Time: 0:15:19 
Iter:   8900,  Train Loss:  0.42,  Train Acc: 87.50%,  Val Loss:  0.77,  Val Acc: 80.17%,  Time: 0:15:30 
Iter:   9000,  Train Loss:  0.38,  Train Acc: 86.72%,  Val Loss:  0.77,  Val Acc: 80.38%,  Time: 0:15:40 
Iter:   9100,  Train Loss:  0.53,  Train Acc: 82.03%,  Val Loss:  0.77,  Val Acc: 80.31%,  Time: 0:15:51 
Iter:   9200,  Train Loss:  0.49,  Train Acc: 82.03%,  Val Loss:   0.7,  Val Acc: 81.77%,  Time: 0:16:02 
Iter:   9300,  Train Loss:  0.47,  Train Acc: 84.38%,  Val Loss:  0.84,  Val Acc: 79.23%,  Time: 0:16:12 
Iter:   9400,  Train Loss:   0.5,  Train Acc: 82.03%,  Val Loss:  0.74,  Val Acc: 80.69%,  Time: 0:16:23 
Iter:   9500,  Train Loss:  0.44,  Train Acc: 85.16%,  Val Loss:  0.63,  Val Acc: 82.71%,  Time: 0:16:33 *
Iter:   9600,  Train Loss:   0.5,  Train Acc: 84.38%,  Val Loss:  0.78,  Val Acc: 80.08%,  Time: 0:16:44 
Iter:   9700,  Train Loss:  0.36,  Train Acc: 86.72%,  Val Loss:  0.67,  Val Acc: 82.39%,  Time: 0:16:55 
Iter:   9800,  Train Loss:  0.32,  Train Acc: 89.06%,  Val Loss:  0.68,  Val Acc: 82.15%,  Time: 0:17:05 
Epoch [8/20]
Iter:   9900,  Train Loss:  0.55,  Train Acc: 82.03%,  Val Loss:   0.8,  Val Acc: 79.22%,  Time: 0:17:16 
Iter:  10000,  Train Loss:  0.45,  Train Acc: 85.94%,  Val Loss:  0.73,  Val Acc: 81.29%,  Time: 0:17:27 
Iter:  10100,  Train Loss:  0.55,  Train Acc: 83.59%,  Val Loss:  0.69,  Val Acc: 81.77%,  Time: 0:17:37 
Iter:  10200,  Train Loss:  0.47,  Train Acc: 85.16%,  Val Loss:  0.64,  Val Acc: 82.91%,  Time: 0:17:48 
Iter:  10300,  Train Loss:  0.49,  Train Acc: 83.59%,  Val Loss:  0.66,  Val Acc: 82.10%,  Time: 0:17:58 
Iter:  10400,  Train Loss:  0.42,  Train Acc: 89.06%,  Val Loss:  0.76,  Val Acc: 80.48%,  Time: 0:18:09 
Iter:  10500,  Train Loss:  0.34,  Train Acc: 88.28%,  Val Loss:  0.69,  Val Acc: 81.89%,  Time: 0:18:20 
Iter:  10600,  Train Loss:  0.36,  Train Acc: 87.50%,  Val Loss:  0.66,  Val Acc: 82.47%,  Time: 0:18:31 
Iter:  10700,  Train Loss:  0.41,  Train Acc: 86.72%,  Val Loss:  0.71,  Val Acc: 81.48%,  Time: 0:18:41 
Iter:  10800,  Train Loss:  0.42,  Train Acc: 86.72%,  Val Loss:  0.72,  Val Acc: 81.07%,  Time: 0:18:52 
Iter:  10900,  Train Loss:  0.45,  Train Acc: 83.59%,  Val Loss:  0.95,  Val Acc: 77.22%,  Time: 0:19:02 
Iter:  11000,  Train Loss:  0.49,  Train Acc: 82.81%,  Val Loss:  0.69,  Val Acc: 81.98%,  Time: 0:19:13 
Iter:  11100,  Train Loss:  0.46,  Train Acc: 89.06%,  Val Loss:  0.74,  Val Acc: 81.03%,  Time: 0:19:24 
Iter:  11200,  Train Loss:  0.46,  Train Acc: 86.72%,  Val Loss:  0.73,  Val Acc: 81.29%,  Time: 0:19:34 
Epoch [9/20]
Iter:  11300,  Train Loss:  0.47,  Train Acc: 85.16%,  Val Loss:  0.66,  Val Acc: 82.31%,  Time: 0:19:45 
Iter:  11400,  Train Loss:  0.39,  Train Acc: 86.72%,  Val Loss:  0.65,  Val Acc: 82.84%,  Time: 0:19:55 
Iter:  11500,  Train Loss:  0.35,  Train Acc: 89.84%,  Val Loss:   0.8,  Val Acc: 80.19%,  Time: 0:20:06 
No optimization for a long time, auto-stopping...
THUCNews/saved_dict/Transformer.ckpt
Test Loss:   0.6,  Test Acc: 83.19%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.7877    0.8610    0.8227      1000
       realty     0.8786    0.8900    0.8843      1000
       stocks     0.7010    0.8230    0.7571      1000
    education     0.9354    0.9270    0.9312      1000
      science     0.7347    0.7670    0.7505      1000
      society     0.9025    0.8610    0.8813      1000
     politics     0.8665    0.8050    0.8346      1000
       sports     0.9985    0.6560    0.7918      1000
         game     0.8207    0.8740    0.8465      1000
entertainment     0.7917    0.8550    0.8221      1000

     accuracy                         0.8319     10000
    macro avg     0.8417    0.8319    0.8322     10000
 weighted avg     0.8417    0.8319    0.8322     10000

Confusion Matrix...
[[861  16  86   4  18   2   8   0   1   4]
 [ 30 890  36   1  14   8   2   0   6  13]
 [ 83  25 823   1  38   1  19   0   6   4]
 [  2   4   9 927   9  20  13   0   3  13]
 [ 27  14  74   9 767  10  16   0  66  17]
 [ 10  28   9  10  21 861  32   0   8  21]
 [ 29  15  57  14  33  25 805   0   7  15]
 [ 35   6  43  10  43  14  28 656  36 129]
 [ 10   9  24   3  65   4   2   0 874   9]
 [  6   6  13  12  36   9   4   1  58 855]]
Time usage: 0:00:02

Process finished with exit code 0
```







