import torch
import torch.nn as nn
import math
from collections import OrderedDict
class Layer(nn.Sequential):
    '''block中的每个卷积层包含三层卷积，先升维，再卷积，再降维'''
    def __init__(self, num_input_features, num_output_features=32, bn_size=4):
        '''维度默认×4'''
        super(Layer, self).__init__()
        #先把维度升上去
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                            num_input_features, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm1', nn.BatchNorm2d(num_input_features*bn_size)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        #作3×3卷积
        self.add_module('conv2', nn.Conv2d(bn_size * num_input_features, bn_size*num_input_features,
                                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        #把维度降下来
        self.add_module('conv3',nn.Conv2d(bn_size*num_input_features,num_output_features,
                                          kernel_size=1,stride=1,bias=False)),
        self.add_module('norm3',nn.BatchNorm2d(num_output_features)),
        self.add_module('relu3',nn.ReLU(inplace=True))

    # def forward(self,x):
    #     new_features=super(Layer,self).forward(x)
    #     return new_features
def Get_Lines(num_layers,drop_rate):
    '''根据层数和drop率计算出一种连接方式'''
    #连线的列表,不包含相邻层之间的连线，因为相邻层之间必须相连
    lines = []
    #除相邻层外，其他层之间所有可能的连线，每条线用tuple表示，比如输入和第5层有连接(0,5)
    for i in range(0, num_layers - 1):
        for j in range(i + 2, num_layers + 1):
            lines.append((i, j))
    #连线总数
    num_lines=0
    #连线总数的计算方式，f(n)=(n-1)+...+1
    for p in range(1,num_layers):
        num_lines+=p
    #全1的张量
    r = torch.ones(num_lines)
    #以drop_rate令某些1变为0
    r = torch.nn.Dropout(drop_rate)(r)
    #根据r从lines中以drop_rate概率去掉一些边，即置0
    for k in range(0, len(lines)):
        if r.data[k] == 0:
            lines[k] = 0
    return lines
def Get_Input_Channels(lines,layer_mark):
    '''根据连边列表和本层序号，统计本层卷积的输入通道数'''
    input_channels=0
    for i in range(len(lines)):
        if lines[i]!=0:
            if lines[i][1]==layer_mark:
                input_channels+=1
    return input_channels
class Block(nn.Sequential):
    def __init__(self,lines,num_layers,num_input_features,bn_size):
        super(Block,self).__init__()
        # self.num_layers=num_layers
        self.lines=lines
        # self.layer_name_list=[]
        #第一层,输入只能来自最初到输入，所以可以直接添加
        self.add_module('layer1',Layer(num_input_features=num_input_features,
                        num_output_features=num_input_features,bn_size=bn_size))
        # self.layer_name_list.append('layer1')
        #循环添加第2到最后一层
        for layer_mark in range(2,num_layers+1):
            #首先计算除了相邻的上一层外，还有前面几层和本层相连，以确定卷积操作输入通道数
            input_channels=Get_Input_Channels(lines,layer_mark=layer_mark)
            layer_name='layer{}'.format(layer_mark)
            self.add_module(layer_name,Layer(num_input_features=num_input_features+input_channels*num_input_features,
                        num_output_features=num_input_features,bn_size=4))
        #保存block每层卷积到输出列表，方便前向传播时从中获取每一层的输入
        self.output_list=[]
    def get_layer_input(self,layer,layer_input):
        '''获取本层输入'''
        for i in range(len(self.lines)):
            #lines中的元素不为0,说明存在该连线
            if self.lines[i]!=0:
                #如果不为0的tuple的第二个数字是本层的层号，则看tuple的第一个数字是几，即前面第几层和本层有连线
                if self.lines[i][1]==layer:
                    #得到与本层连接的层号，再按层号到out_put_list中索引到那一层的输出，与本层的原始输入进行cat
                    layer_input=torch.cat([layer_input,self.output_list[self.lines[i][0]]],1)
        return layer_input
    def forward(self,x):
        out0=x
        out1=self.layer1(x)
        self.output_list.append(out0)
        self.output_list.append(out1)
        layer_input=self.get_layer_input(2,out1)
        out2=self.layer2(layer_input)
        self.output_list.append(out2)
        layer_input=self.get_layer_input(3,out2)
        out3=self.layer3(layer_input)
        self.output_list.append(out3)
        layer_input=self.get_layer_input(4,out3)
        out4=self.layer4(layer_input)
        self.output_list.append(out4)
        layer_input=self.get_layer_input(5,out4)
        out5=self.layer5(layer_input)
        return out5
        # for layer in range(2,self.num_layers+1):
        #     layer_input=self.get_layer_input(layer=layer,layer_input=layer_input)
        #     out=self.
class Transitioin(nn.Sequential):
    '''block中间进行转换，通道数×2,特征图大小/2'''
    def __init__(self,num_input_features,factor=2):
        super(Transitioin,self).__init__()
        self.add_module('conv1×1',nn.Conv2d(in_channels=num_input_features,out_channels=num_input_features*factor,
                                            kernel_size=1,stride=1,padding=0,bias=False))
        self.add_module('bn',nn.BatchNorm2d(num_features=num_input_features*factor))
        self.add_module('relu',nn.ReLU(True))
        self.add_module('pool',nn.MaxPool2d(kernel_size=2,stride=2,padding=0))

class Random_Connect(nn.Module):
    '''网络结构'''
    def __init__(self,num_init_features=32,bn_size=4,num_layers=5,drop_rate=0.4,num_classes=10):
        '''受block中前向传播的影响，每个block的层数必须固定，默认是5'''
        super(Random_Connect,self).__init__()

        # self.num_input_features=num_init_features
        #维度升到num_init_features
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=1, stride=1, padding=0, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        #
        lines = Get_Lines(num_layers, drop_rate)
        print(lines)
        #第一个block每个卷积层均为32通道，特征他均为32×32
        self.features.add_module('block1',Block(lines=lines,num_layers=num_layers,num_input_features=num_init_features,bn_size=bn_size))
        #经过一个transition层，通道×2,特征图大小/2
        self.features.add_module('trans1',Transitioin(num_input_features=num_init_features,factor=2))
        lines = Get_Lines(num_layers, drop_rate)
        print(lines)
        #第二个block每个卷积层均为64通道，特征图均为16×16
        self.features.add_module('block2',Block(lines=lines,num_layers=num_layers,num_input_features=num_init_features*2,bn_size=bn_size))
        #全局均值池化
        self.avg_pool=nn.AvgPool2d(16,1,0)
        #线性层，第二个block的输出是64通道，全局均值池化后为batch_size*64
        self.fc=nn.Linear(64,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        x=self.features(x)

        x=self.avg_pool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)


        return x