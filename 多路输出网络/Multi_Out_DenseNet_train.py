import torch
from torch import nn,optim
from torchvision import datasets ,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Multi_Out_DenseNet import DenseNet

#超参数
batch_size=50
learning_rate=1e-1
num_epoches=10

#数据集加载
tf=transforms.Compose([transforms.ToTensor()])
train_dataset=datasets.CIFAR10('./data',train=True,transform=tf,download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

#模型声明
model=DenseNet(growth_rate=8,block_config=(5,5,5),num_init_features=8,drop_rate=0.2,num_classes=10)
# model.load_state_dict(torch.load('Simplified_DenseNet_of_epoches15.pkl'))
if torch.cuda.is_available():
    model=model.cuda()
# print(model.block1.parameters())

# for pram in model.named_parameters():
#     print(pram[0])
# print('///////////////////')
# for pram in model.block1.named_parameters():
#     print(pram[0])
# exit()

#损失函数及优化器
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD([{"params":model.block1.parameters()},
                        {"params":model.fc1.parameters()}], lr=learning_rate)
optimizer2 = optim.SGD([{"params":model.block2.parameters()},
                        {"params":model.fc2.parameters()},
                        {"params":model.trans1.parameters()}], lr=learning_rate)
optimizer3 = optim.SGD([{"params":model.block3.parameters()},
                        {"params":model.fc3.parameters()},
                        {"params":model.trans2.parameters()}], lr=learning_rate)
# 学习率衰减，每10个epoch乘以0.2
learning_rate_decay = optim.lr_scheduler.StepLR(optimizer1, 5, 0.1)

#记录损失值，方便写入csv文件
# import csv
# loss_data=[]
# file=open('Simplified_DenseNet.csv','a')
# writer=csv.writer(file)

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
        out1,out2,out3 = model(inputs)
        # print(out1.data.size())
        # print(out2.data.size())
        # print(out3.data.size())
        # exit()
        loss1 = criterion(out1, labels)
        loss2=criterion(out2,labels)
        loss3=criterion(out3,labels)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward(retain_graph=True)

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        print('epoch/batch->%2d/%4d........loss1/loss2/loss3->%.4f %.4f %.4f' % (epoch, i, loss1.data[0],loss2.data[0],loss3.data[0]))

#         #每个epoch产生1000个loss,共15000个loss值,取每50个的第一个，共取300个
#         if i%50==0:
#             loss_data.append(str(loss.data[0]))
#             # writer.writerow(loss_data)
#             # loss_data.clear()
#     # if(epoch+1)%5==0:
#         #每5个epoch保存一个模型
#         # torch.save(model.state_dict(), './Simplified_DenseNet_of_epoches{}.pkl'.format(epoch+1))
# writer.writerow(loss_data)
# file.close()
# # torch.save(model.state_dict(),'./Simplified_DenseNet.pkl')