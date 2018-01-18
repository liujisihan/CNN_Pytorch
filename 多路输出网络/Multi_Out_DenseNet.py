#修改版的DenseNet,3个block，每个block5次卷积，每个block的输出都接一个fc，3路输出，分别训练
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    '''直接借用DenseNet的layer层'''
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.1', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))

class Fc(nn.Sequential):
    def __init__(self,in_features):
        super(Fc,self).__init__()
        self.add_module("linear1",nn.Linear(in_features,in_features*2))
        self.add_module("batch_norm1",nn.BatchNorm1d(in_features*2))
        self.add_module('drop_out1',nn.Dropout(0.2))
        self.add_module("relu1",nn.ReLU(True))

        self.add_module('linear2',nn.Linear(in_features*2,in_features*2))
        self.add_module("batch_norm1", nn.BatchNorm1d(in_features * 2))
        self.add_module('drop_out2',nn.Dropout(0.2))
        self.add_module("relu1", nn.ReLU(True))

        self.add_module('linear3',nn.Linear(in_features*2,10))

    def forward(self,x):
        x=nn.AvgPool2d(x.size(2))(x)
        x=x.view(x.size(0),-1)
        # print("fc input:",x.data.size())
        x=super(Fc,self).forward(x)

        return x
class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=8, block_config=(5,5,5),
                 num_init_features=8, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution  进入block之前先把维度升到num_init_features=8
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        #第一block，5层，输入是num_init_features=8通道，内部维度升高2倍，输出8通道，drop_rate=0.2
        self.block1=_DenseBlock(5,num_init_features,2,8,0.2)#[50,48,32,32]
        self.trans1=_Transition(48,24)#[50,24,16,16]
        self.block2=_DenseBlock(5,24,2,8,0.2)#[50,64,16,16]
        self.trans2=_Transition(64,32)#[50,32,8,8]
        self.block3=_DenseBlock(5,32,2,8,0.2)#[50,72,8,8]

        self.fc1=Fc(48)
        self.fc2=Fc(64)
        self.fc3=Fc(72)
        # Each denseblock
        # num_features = num_init_features
        # for i, num_layers in enumerate(block_config):
        #     block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
        #                         bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        #     self.features.add_module('denseblock%d' % (i + 1), block)
        #     num_features = num_features + num_layers * growth_rate
        #     if i != len(block_config) - 1:
        #         trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        #         self.features.add_module('transition%d' % (i + 1), trans)
        #         num_features = num_features // 2
        #
        # # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        #
        # # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x=self.features(x)

        out1=self.block1(x)
        # print("block1 output:",out1.data.size())
        fc1_out=self.fc1(out1)
        # print("fc1 output:",fc1_out.data.size())

        out2=self.block2(self.trans1(out1))
        # print("block2 output:", out2.data.size())
        fc2_out=self.fc2(out2)
        # print("fc2 output:",fc2_out.data.size())

        out3=self.block3(self.trans2(out2))
        # print("block3 output:", out3.data.size())
        fc3_out=self.fc3(out3)
        # print("fc3 output:",fc3_out.data.size())
        # exit()

        # features = self.features(x)
        # out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        # out = self.classifier(out)
        return fc1_out,fc2_out,fc3_out