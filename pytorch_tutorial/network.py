import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same convolution
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#14*14
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1=nn.Linear(16*7*7,num_classes)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)#64*784
        x=self.fc1(x)
        return x

#set device
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
#hyperparameters
in_channel=1
input_size=784
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=1
#Load data
train_dataset=datasets.MNIST(download=False,root='./dataset/',train=True,transform=transforms.ToTensor())
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset=datasets.MNIST(download=False,root='./dataset/',train=False,transform=transforms.ToTensor())
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
#Initialize netwark
# model=NN(input_size=input_size,num_classes=num_classes).to(device)
model=CNN(in_channels=in_channel,num_classes=num_classes).to(device)
#Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
# for batch, (data, targets) in enumerate(train_loader):#1个batch里64个数据
#     print(targets.shape )
#train network
for epoch in range(num_epochs):
    for batch, (data,targets) in enumerate(train_loader):
        data=data.to(device=device)
        targets=targets.to(device=device)
        # #get to correct shape
        # data=data.reshape(data.shape[0],-1)#64*784
        #forward
        scores=model(data)#64*10
        loss =criterion(scores,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent
        optimizer.step()
#check the accuracy
def check_accuracy(loader,model):
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            # x=x.reshape(x.shape[0],-1)
            scores=model(x)#64*10
            # scores.max()
            _,prediction=scores.max(dim=1)#返回的是最大可能类别标签的索引
            num_correct+=(prediction==y).sum()
            num_samples+=prediction.size(0)
        print(f'got{num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} in {loader}')
    model.train()
check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

