from Simplified_DenseNet import DenseNet
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 超参数
batch_size=50

# 图像预处理
data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

# 加载测试集

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf, download=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model=DenseNet(growth_rate=32,block_config=[5],num_init_features=32,bn_size=4,drop_rate=0.2,num_classes=10)
model.load_state_dict(torch.load('./Simplified_DenseNet_1/Simplified_DenseNet_15.pkl'))
if torch.cuda.is_available():
    model=model.cuda()

# 损失及优化器，这里只需要损失函数
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 测试损失值和精确度
test_loss=0.0
test_acc=0.0

# 模型变为测试模式
model.eval()

# 测试
for i,data in enumerate(test_loader,0):
    inputs,labels=data
    if torch.cuda.is_available():
        # volatile=True表示前向传播时不保留缓存，用于测试模型，节省内存
        inputs=Variable(inputs,volatile=True).cuda()
        labels=Variable(labels,volatile=True).cuda()
    else:
        inputs = Variable(inputs, volatile=True)
        labels = Variable(labels, volatile=True)

    out=model.forward(inputs)
    loss=criterion(out,labels)
    test_loss+=loss.data[0]*labels.size(0)
    _,predict=torch.max(out,1)
    num_correct=(predict==labels).sum()
    test_acc+=num_correct.data[0]
    print('batch of {}/200'.format(i))

print('Test Loss:{:.6f},Accuracy:{:.6f}'.format(test_loss/(len(test_dataset)),test_acc/(len(test_dataset))))
#Test Loss:0.733816,Accuracy:0.740800