# CNN+global avg pool模型，三层卷积，进行训练
import torch
import math
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import CNN_with_gap
import csv
import time

# 超参数
batch_size = 32
learning_rate = 1e-2
num_epoches = 30

# 图像预处理
data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

# 加载数据集
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=data_tf, download=True)
#test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=data_tf, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 网络参数初始化
def Initialize_Parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_()

# 网络声明
model = CNN_with_gap.CNN_with_gap()
if torch.cuda.is_available():
    model = model.cuda()

# 参数初始化
Initialize_Parameters(model)

# 损失及优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 学习率衰减，每10个epoch乘以0.1
learning_rate_decay = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

#记录损失值和步数，方便绘图，每20个batch记录一次，loss_data的值是20个batch的平均值
loss_data=0.0
#steps=0步数可以不用记录，绘图时标注“×20”

#记录所有损失值，写入文件,方便以后绘图
loss_list=[]

#记录训练时间
start=time.time()

# 训练
for epoch in range(num_epoches):
    learning_rate_decay.step()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        out = model.forward(inputs)

        loss = criterion(out, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        print('epoch/batch->%2d/%4d........%s->%.5f' % (epoch, i, 'loss', loss.data[0]))

        loss_data+=loss.data[0]
        if (i+1) % 20 ==0:
            loss_data/=20.0
            loss_list.append(loss_data)
            loss_data=0.0

    ##一个epoch结束，需把loss_data置0，因为batch数量不能整除20，在epoch结尾的几个
    #batch的loss值会累加到下个epoch的前二十个batch的loss上
    loss_data=0.0

#训练时间
end=time.time()
train_time=end-start

#写入csv文件
file=open('CNN_with_gap_loss.csv','a')
writer=csv.writer(file)
writer.writerow(['train time',train_time])
writer.writerow(loss_list)
file.close()

torch.save(model.state_dict(),'./CNN_with_gap.pkl')
