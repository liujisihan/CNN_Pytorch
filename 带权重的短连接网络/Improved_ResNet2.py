import torch.nn as nn
import math
import torch
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):


    def __init__(self, in_features):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_features//4)
        self.conv2 = nn.Conv2d(in_features//4, in_features//4, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_features//4)
        self.conv3 = nn.Conv2d(in_features//4,in_features, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)

        # self.drop2d=nn.Dropout2d(0.5)


    def forward(self, x):
        self.avg_pool = nn.AvgPool2d(x.data.size(2),1)


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out_weight=self.avg_pool(out)
        # # print('out_weight:',out_weight.data.size())
        # x_weight=self.avg_pool(x)
        # # print('x_weight:',x_weight.data.size())
        # add=out_weight+x_weight
        # out_weight=out_weight/add
        # # print('out_weight2',out_weight.data.size())
        # x_weight=x_weight/add
        # # print('x_weight2',x_weight.data.size())
        # out = out * out_weight
        # # print('out:',out.data.size())
        # x = x * x_weight
        # # print('x:',x.data.size())
        # # exit()
        # out += x

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #     out=self.downsample(out)
        # zero_or_one=torch.randperm(2)[0]
        # zero_or_one=Variable(torch.cuda.FloatTensor([zero_or_one]),requires_grad=False)
        # x=x*zero_or_one
        x=x*0.2
        out+=x
        out = self.relu(out)

        return out

class Transition(nn.Sequential):
    def __init__(self,in_features):
        super(Transition,self).__init__()
        self.add_module("transition_conv",nn.Conv2d(in_features,in_features*2,1,1))
        self.add_module("transition_pool",nn.MaxPool2d(2,2))
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):#初始化，block类型，层安排，类别数
        #self.inplanes = 64#默认输入16维
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)#把维度升到16维
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],1)#[50,128,16,16]
        self.layer2 = self._make_layer(block, 128, layers[1],1)#[50,256,8,8]
        self.layer3 = self._make_layer(block, 256, layers[2],0)#[50,256,8,8]
        #self.layer4 = self._make_layer(block, 512, layers[3])
        # self.bn2=nn.BatchNorm2d(256)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(256 , num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks,transition=0):#block类型，输入通道，block个数，步长
        layers = []
        for i in range(0, blocks):
            layers.append(block(planes))
        if transition==1:
            layers.append(Transition(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x=self.bn2(x)
        # x=self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x