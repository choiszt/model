import torch
batch_size=10
features=25
x=torch.rand((batch_size,features))

print(x[:,0])
x=torch.arange(11)
print(torch.where(x>5),x,x*2)
