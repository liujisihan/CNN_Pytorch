import CNN_with_fc
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 超参数
batch_size=32

# 图像预处理
data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

# 加载测试集
#train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=data_tf, download=True)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model=CNN_with_fc.CNN_with_fc()
model.load_state_dict(torch.load('CNN_with_fc.pkl'))
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

print('Test Loss:{:.6f},Accuracy:{:.6f}'.format(test_loss/(len(test_dataset)),test_acc/(len(test_dataset))))
# Test Loss:0.077986,Accuracy:0.977100