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

