import torch
from torch import nn,optim
from torchvision import datasets ,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Simplified_DenseNet import DenseNet

#超参数
batch_size=50
learning_rate=1e-1
num_epoches=10

#数据集加载
tf=transforms.Compose([transforms.ToTensor()])
train_dataset=datasets.CIFAR10('./data',train=True,transform=tf,download=False)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

#模型声明
model=DenseNet(growth_rate=32,block_config=(5,5,5),num_init_features=32,drop_rate=0.2,num_classes=10)
# model.load_state_dict(torch.load('Simplified_DenseNet_of_epoches15.pkl'))
if torch.cuda.is_available():
    model=model.cuda()
# print(model)
# exit()

#损失函数及优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 学习率衰减，每10个epoch乘以0.2
learning_rate_decay = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

#记录损失值，方便写入csv文件
import csv
loss_data=[]
file=open('Simplified_DenseNet.csv','a')
writer=csv.writer(file)

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
        print('epoch/batch->%2d/%4d........%s->%.5f' % (epoch, i, 'loss',loss.data[0]))

        #每个epoch产生1000个loss,共15000个loss值,取每50个的第一个，共取300个
        if i%50==0:
            loss_data.append(str(loss.data[0]))
            # writer.writerow(loss_data)
            # loss_data.clear()
    # if(epoch+1)%5==0:
        #每5个epoch保存一个模型
        # torch.save(model.state_dict(), './Simplified_DenseNet_of_epoches{}.pkl'.format(epoch+1))
writer.writerow(loss_data)
file.close()
# torch.save(model.state_dict(),'./Simplified_DenseNet.pkl')