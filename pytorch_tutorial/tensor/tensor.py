import torch

device='cuda' if torch.cuda.is_available() else "cpu"

my_tensor=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,
                       device=device,requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.requires_grad)

#other common initialization methods
x=torch.empty(size=(3,3))
x=torch.zeros((3,3))
x=torch.rand((3,3))
x=torch.ones(3,3)
x=torch.eye(4,5) #I 单位矩阵
x=torch.arange(start=0,end=6,step=1)
x=torch.linspace(start=0.1,end=1,steps=10)
x=torch.empty(size=(1,5)).normal_(mean=0,std=1)
x=torch.empty(size=(1,5)).uniform_(0,1)
x=torch.diag(torch.ones(5))
print(x)

#How to initialize and convert tensors to other types(int,float,double)
tensor=torch.arange(4)
print(tensor.bool())
print(tensor.short())#int16
print(tensor.long())#int64
print(tensor.half())#float16
print(tensor.float())#float32
print(tensor.double())#float64

#array to tensor conversion and vice-versa
import numpy as np
np_array=np.zeros((5,5))
tensor=torch.from_numpy(np_array)
np_array_back=tensor.numpy()
