import torch
x=torch.tensor([1,2,5])
y=torch.tensor([9,8,7])

#addition
z1=torch.empty(3)
torch.add(x,y,out=z1)
print(z1)

z2=torch.add(x,y)
z=x+y
# subtraction
z=x-y
# division
z=torch.true_divide(x,y)
print(z)

#inplace operation
t=torch.zeros(3)#2004589365120
print(id(t))
q=t.add_(x) #2004589365120 加下划线后t和q共用地址
print(id(q))

x1=torch.rand(2,5)
x2=torch.rand(5,2)
x3=torch.mm(x1,x2)
x3=x1.mm(x2)
print(x3)

#matrix exponentation
matrix_exp=torch.rand(5,5)
print(matrix_exp.matrix_power(3))

#element wise mult
z=x*y#分别相乘
print(z)

#dot product
z=torch.dot(x,y)
print(z)
#match matrix multiplication
batch=32
n=10
m=20
p=30
tensor1=torch.rand((batch,n,m))
tensor2=torch.rand((batch,m,p))
out_bmm=torch.bmm(tensor1,tensor2)
print(out_bmm.size())#size(batch,n,p)


a = torch.tensor([[[1, 2, 3, 3.5], [4, 5, 6, 6.5]],
[[7, 8, 9, 9.5], [10, 11, 12, 12.5]],
[[13, 14, 15, 15.5], [16, 17, 18, 18.5]]])
print(a.size()) #torch.Size([3, 2, 4])

#example of broadcasting
x1=torch.rand(10,5)
x2=torch.rand(1,5)

z=x1-x2         #x1的每一行都需要-x2
z=x1**x2

#other useful tensor operations
sum_x=torch.sum(x,dim=0)
values,indices=torch.max(x,dim=0)  #values最大的值 indices索引位置
values,indices=torch.min(x,dim=0)
abs_x=torch.abs(x)
z=torch.argmax(x,dim=0)
print(z)
mean_x=torch.mean(x.float())
print(mean_x)
z=torch.eq(x,y)
print(z)
new=torch.sort(y,dim=0,descending=False)
z=torch.clamp(x,min=4) #小于4的值用4代替

x=torch.tensor([1,0,1,1,1],dtype=bool)
z=torch.any(x)
print(z)
z=torch.all((x))
print(z)