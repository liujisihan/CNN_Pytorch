import torch
from torch import nn,optim
from torchvision import datasets ,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Improved_ResNet2 import ResNet,Bottleneck

#超参数
batch_size=50
learning_rate=1e-1
num_epoches=30

#数据集加载
tf=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_dataset=datasets.CIFAR10('./data',train=True,transform=tf,download=False)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

#模型声明
model=ResNet(Bottleneck,[3,4,6],num_classes=10)
if torch.cuda.is_available():
    model=model.cuda()
# print(model)
# exit()

#损失函数及优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# 学习率衰减，每10个epoch乘以0.1
learning_rate_decay = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

#记录损失值，方便写入csv文件
import csv
loss_data=[]
file=open('Improved_ResNet2.csv','a')
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
        out = model(inputs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch/batch->%2d/%4d........loss->%.4f' % (epoch, i, loss.data[0]))

        #每个epoch产生1000个loss,共30000个loss值,取每50个的第一个，共取600个
        if i%50==0:
            loss_data.append(str(loss.data[0]))
            # writer.writerow(loss_data)
            # loss_data.clear()
    if(epoch+1)%10==0:
        # 每10个epoch保存一个模型
        torch.save(model.state_dict(), './Improved_ResNet2_{}.pkl'.format(epoch+1))
writer.writerow(loss_data)
file.close()
# # torch.save(model.state_dict(),'./Simplified_DenseNet.pkl')