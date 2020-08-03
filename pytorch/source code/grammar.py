import torch
import numpy

a = torch.arange(10)
print(a)
print(a.type())
print(a.shape)
# print(a.size) 打印的是方法信息 built-in method size of tensor object......
print(a.size())
#a_val = a.item() #ValueError: only one element tensors can be converted to Python scalars
a_numpy = a.numpy()
print(a_numpy)

