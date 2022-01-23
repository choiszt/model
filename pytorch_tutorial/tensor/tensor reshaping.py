import torch
x=torch.arange(9)
x_3x3=x.view(3,3)#存储空间地址必须连续
x_3x3=x.reshape(3,3)#存储地址无需连续，另辟空间
print(x_3x3)

y=x_3x3.t()
print(y.contiguous().view(9))

x1=torch.rand(2,5)
x2=torch.rand(2,5)
print(torch.cat((x1,x2),dim=0))
print(torch.cat((x1,x2),dim=1))

z=x1.view(-1) #铺平
print(z)
batch=64
x=torch.rand((batch,2,5))
z=x.view(batch,-1)
print(z.shape)#64*10

z=x.permute(2,0,1)#torch.Size([5, 64, 2]) permute括号里内容为索引
print(z.shape)

x=torch.arange(10,1)
print(x.unsqueeze(1).unsqueeze(0).unsqueeze(0).shape)

z=x.squeeze(0)
print(z)