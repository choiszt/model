import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
class doubleconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(doubleconv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),#bias=False可防止Batchnorm2d对bias的更新
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),  # bias=False可防止Batchnorm2d对bias的更新
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(
            self,in_channels=3,out_channels=1,features=[64,128,256,512],
    ):
        super(Unet,self).__init__()
        self.ups =nn.ModuleList()
        self.downs =nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)

        #down part of unet
        for feature in features:
            self.downs.append(doubleconv(in_channels,feature))
            in_channels=feature

        #up part of unet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2,
                )
            )
            self.ups.append(doubleconv(feature*2,feature))


        self.bottleneck=doubleconv(features[-1],features[-1]*2)
        self.final_conv=nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections=[]
        for down in self.downs:
            x =down(x)
            skip_connections.append(x)
            x=self.pool(x)
        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1]#reverse the list

        for idx in range(0,len(self.ups),2):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//2]
            if x.shape !=skip_connection.shape:
                x=tf.resize(x,size=skip_connection.shape[2:])

            concat_skip=torch.cat((skip_connection,x),dim=1)
            x=self.ups[idx+1](concat_skip)
        return self.final_conv(x)
def new():
    x=torch.randn((3,1,160,160))
    model=Unet(in_channels=1,out_channels=1)
    preds=model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape==x.shape