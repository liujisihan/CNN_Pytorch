#CNN+fc模型，三层卷积，四层全连接

from torch import nn

#全连接神经网络
class FC_Network(nn.Module):
    def __init__(self,in_channels,hidden1_channels,hidden2_channels,num_class=10):
        super(FC_Network,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_channels,hidden1_channels),nn.BatchNorm1d(hidden1_channels),nn.Dropout(0.5),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden1_channels,hidden2_channels), nn.BatchNorm1d(hidden2_channels),nn.Dropout(0.5),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(hidden2_channels,num_class))
    def forward(self, x):
        x=x.view(x.size(0),-1)
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        return out

#定义网络结构
class CNN_with_fc(nn.Module):
    def __init__(self):
        super(CNN_with_fc,self).__init__()
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
        layer3.add_module('conv3',nn.Conv2d(32,64,3,1,padding=1))#输出是32*64*14*14
        layer3.add_module('bn3',nn.BatchNorm2d(64))
        layer3.add_module('relu3',nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))#输出是32*64*7*7
        self.layer3=layer3
        self.fc=FC_Network(64*7*7,512,128,10)

    def forward(self, x):
        x=self.layer1(x)#第一层卷积输出
        x=self.layer2(x)#第二卷积层输出
        x=self.layer3(x)# 第三卷积层的输出
        out=self.fc.forward(x)
        return out