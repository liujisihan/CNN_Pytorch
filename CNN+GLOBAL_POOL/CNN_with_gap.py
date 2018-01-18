#CNN+global avg pool模型，三层卷积

from torch import nn
import torch
#定义网络结构
class CNN_with_gap(nn.Module):
    def __init__(self):
        super(CNN_with_gap,self).__init__()
        #第一卷积层
        layer1=nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(1, 16, 3, 1, padding=1))  # 输入是32*1*28*28，out:32*16*28*28
        layer1.add_module('bn1', nn.BatchNorm2d(16))
        layer1.add_module('relu1', nn.ReLU(True))  # True表示原地修改，不维护影子变量
        #layer1.add_module('pool1', nn.MaxPool2d(2, 2))  # out:32,16,14,14
        self.layer1 = layer1

        #第二卷积层
        layer2=nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(16,32,3,1,padding=1))#输出是32*32*28*28
        layer2.add_module('bn2',nn.BatchNorm2d(32))
        layer2.add_module('relu2',nn.ReLU(True))
        layer2.add_module('pool2',nn.MaxPool2d(2,2))#输出32*32*14*14
        self.layer2=layer2

        #第三卷积层
        layer3=nn.Sequential()
        layer3.add_module('conv3',nn.Conv2d(32,10,3,1,padding=1))#输出是32*10*14*14
        layer3.add_module('bn3',nn.BatchNorm2d(10))
        layer3.add_module('relu3',nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))#输出是32*10*7*7
        self.layer3=layer3

        self.global_avg_pool=nn.AvgPool2d(7)

    def forward(self, x):
        x=self.layer1(x)#第一层卷积输出
        avg_feature=torch.sum(x,1)/x.data.size(1)

        x=self.layer2(x+avg_feature)#第二卷积层输出
        print(x.data.size())
        exit()
        x=self.layer3(x)# 第三卷积层的输出
        out=self.global_avg_pool(x)
        print(out.data.size())

        out=out.view(out.size(0),10)#全局均值池化后32*10*1*1,需要reshape成32*10

        return out