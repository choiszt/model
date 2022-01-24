import torch
import torch.nn as nn
VGG16=[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
#then flatten and 4096x4096x1000 Linear layers

class VGG(nn.Module):
    def __init__(self,in_channels=3,num_classes=1000):
        super(VGG,self).__init__()
        self.in_channels=in_channels
        self.conv_layers=self.create_conv_layers(VGG16)
        self.fc=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x=self.conv_layers(x)
        # self.shape1=x.shape
        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        # return self.shape1
        return x
    def create_conv_layers(self,architecture):
        layers=[]
        in_channels=self.in_channels
        for x in architecture:
            if type(x)==int:
                out_channels =x

                layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),
                                   stride=(1,1),padding=(1,1)),nn.BatchNorm2d(x),nn.ReLU()]
                self.layers=layers
                in_channels=x
            elif x=='M':
                layers+=[nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
        return nn.Sequential(*layers)

model=VGG(3,1000)
x=torch.rand(1,3,244,244)
print(model(x))