#import torch
#
#cpu_tensor = torch.zeros(2,3)
#device = torch.device("cuda:0")
#gpu_tensor = cpu_tensor.to(device)
#print(gpu_tensor)

#-------------------------------------
# 3장 선형 회귀 분석.
#-------------------------------------

#import torch
#
#x = torch.tensor(data=[2.0, 3.0], requires_grad=True)
#y = x**2
#z = 2*y + 3
#
#target = torch.tensor([3.0, 4.0])
#loss = torch.sum(torch.abs(z-target))
#loss.backward()
#
#print(x.grad, y.grad, z.grad)

#-------------------------------------
#import torch
#import torch.nn as nn           #신경망 모델 중 Linear 함수
#import torch.optim as optim     #경사하강법 알고리즘 
#import torch.nn.init as init    #텐서 초기값을 주기위한 필요 함수
#
#
#num_data=1000 #데이터 수
#num_epoch=500 #선형회귀반복횟수
#
#x = init.uniform_(torch.Tensor(num_data,1),-10,10) #-10, 10의 균등한 데이터
#noise=init.normal_(torch.FloatTensor(num_data,1),std=1) #가우시안 노이즈
#
#y=2*x+3 #일반 함수 (y=2x+3)
#y_noise=2*(x+noise)+3 #노이즈 추가된 함수
#
##선형 회귀 모델
#model=nn.Linear(1,1) #들어오는 특성, 결과로 나오는 특성의 수, 편차 사용 여부
#loss_func=nn.L1Loss() #L1손실 : 차이의 절댓값의 평균. loss(x,y)=(1/n)Σ|x_i-y_i|
#
##최적화 함수. 경사하강법을 적용하여 오차를 줄이고 최적의 가중치와 편차를 근사할 수 있게하는 역할.
##SGD(stochastic gradient descent) 한번에 들어오는 데이터의 수대로 경사하강법 알고리즘을 적용.
##최적화할 변수들과 함께 학습률을 lr이라는 인수로 전달.
##model.parameters()로  선형회구모델의 변수 w와 b를 전달. (가중치 : weight, 편차 : bias)
#optimizer = optim.SGD(model.parameters(),lr=0.01)
#
#label = y_noise
#for i in range(num_epoch):
#    optimizer.zero_grad()        #이전 스텝에서 계산한 기울기 0으로 초기화.
#    output = model(x)              #선형회귀모델에 값 저장.
#
#    loss = loss_func(output,label) #output과 y_noise의 차이를 loss에 저장.
#    loss.backward()              #w, b에 대한 기울기가 계산됨.
#    #인수로 들어갔던 model.parameters()에서 리턴되는 변수들의 기울기에 학습률 0.01을 곱하여 빼줌으로써 업데이트.
#    optimizer.step()
#
#    # 10번에 한번씩 손실률 Display.
#    if i % 10 == 0:
#        print(loss.data)
#
#param_list=list(model.parameters())
#print(param_list[0].item(),param_list[1].item())
#-------------------------------------
# 4장 인공 신경망.
#-------------------------------------

#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.init as init
#
#num_data = 1000
#num_epoch = 10000
#
#noise = init.normal_(torch.FloatTensor(num_data, 1),std=1)
#x = init.uniform_(torch.Tensor(num_data, 1), -15, 15)
#y = (x**2) + 3
#y_noise = y + noise
#
#
#model = nn.Sequential(
#    nn.Linear(1,6),
#    nn.ReLU(),
#    nn.Linear(6,10),
#    nn.ReLU(),
#    nn.Linear(10,6),
#    nn.ReLU(),
#    nn.Linear(6,1),
#    )
#
#loss_func = nn.L1Loss()
#optimizer = optim.SGD(model.parameters(), lr=0.0002)
#
#loss_array = []
#for i in range(num_epoch):
#    optimizer.zero_grad()
#    output = model(x)
#
#    loss = loss_func(output, y_noise)
#    loss.backward()
#    optimizer.step()
#
#    loss_array.append(loss)
#
#import matplotlib.pyplot as plt
#
#plt.plot(loss_array)
#plt.show()

#-------------------------------------
# 5장 합성곱 신경망.
#-------------------------------------
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.init as init
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
#
#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN,self).__init__() # super클래스는 CNN 클래스의 부모클래스인 nn.Module을 초기화 하는 역할.
#        self.layer = nn.Sequential(
#            nn.Conv2d(1,16,5),
#            nn.ReLU(),
#            nn.Conv2d(16,32,5),
#            nn.ReLU(),
#            nn.MaxPool2d(2,2),
#            nn.Conv2d(32,64,4),
#            nn.ReLU(),
#            nn.MaxPool2d(2,2)
#            )
#        self.fc_layer = nn.Sequential(
#            nn.Linear(64*3*3, 100),
#            nn.ReLU(),
#            nn.Linear(100, 10)
#            )
#
#    def forward(self,x):
#        out = self.layer(x)
#        out = out.view(batch_size, -1)
#        out = self.fc_layer(out)
#        return out
#
#batch_size = 256
#learning_rate = 0.0002
#num_epoch = 10
#
#mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(),
#                        target_transform=None, download=True)
#mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(),
#                        target_transform=None, download=True)
#
#train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, 
#                                           shuffle=True, num_workers=0, drop_last=True)
#test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, 
#                                           shuffle=False, num_workers=0, drop_last=True)
#
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = CNN().to(device)
#loss_func = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#loss_arr = []
#for i in range(num_epoch):
#    for j,[image,label] in enumerate(train_loader):
#        x = image.to(device)
#        y_= label.to(device)
#
#        optimizer.zero_grad()
#        output = model.forward(x)
#        loss = loss_func(output, y_)
#        loss.backward()
#        optimizer.step()
#
#        if j % 1000 == 0:
#            print(loss)
#            loss_arr.append(loss.cpu().detach().numpy())
#
#correct = 0
#total = 0
#
#with torch.no_grad():
#    for image, label in test_loader:
#        x = image.to(device)
#        y_= label.to(device)
#
#        ouput          = model.forward(x)
#        _,output_index = torch.max(ouput,1)
#
#        total   += label.size(0)
#        correct += (output_index == y_).sum().float()
#    print("Accuracy of Test Data: {}".format(100 * correct / total))

#--------------------------------------------------------------------------------------
#import torch
#import torchvision
#import torchvision.transforms as tr
#from torch.utils.data import DataLoader, Dataset
#import numpy as np
#import CovModule as cm
#
#transf = tr.Compose([tr.Resize(8), tr.ToTensor()])
#
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transf)
#
#trainset[0][0].size()
#
#trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
#testloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
#
#len(trainloader)
#
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
#images.size()
#
#
#transf = tr.Compose([tr.Resize(16), tr.ToTensor()])
#trainset1 = torchvision.datasets.ImageFolder(root='./class', transform=transf)
#trainloader1 = DataLoader(trainset, batch_size=10, shuffle=False, num_workers=2)
#
#
#class TensorData(Dataset):
#    def __init__(self, x_data, y_data):
#        self.x_data = torch.FloatTensor(x_data)
#        self.x_data = self.x_data.permute(0,3,1,2)
#        self.y_data = torch.longTensor(y_data)
#        self.len = self.Y_data.shape[0]
#
#    def __getitem__(self, index):
#        return self.x_data[index], self.y_data[index]
#
#    def __len__(self):
#        return self.len
#
#train_data = TensorData(train_images, train_labels)
#train_loader = DataLoader(train_data, batch_size=10, shuffle=False, num_workers=2)
#
#trans = tr.Compose([cm.ToTensor(), cm.LinearTensor(2,5)])
#ds1 = cm.MyDataset(train_images, train_labels, transform=trans)
#train_loader1 = DataLoader(ds1, batch_size=10, shuffle=True)


#--------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load Data set
transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                          shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Build a model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        self.fc_layer = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
            )
        #self.conv1 = nn.Conv2d(3,6,5)
        #self.pool = nn.MaxPool2d(2,2)
        #self.conv2 = nn.Conv2d(6,16,5)
        #self.fc1 = nn.Linear(16*5*5, 120)
        #self.fc2 = nn.Linear(120,84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1,16 * 5 * 5)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x
        out = self.layer(x)
        out = out.view(-1,16 * 5 * 5)
        out = self.fc_layer(out)
        return out

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
# CPU
#net = Net()

print('%s' % device)

# Implement the model with training set
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)


#for epoch in range(1): # loop over the dataset muliple times
min = 2.0
for epoch in range(20):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data;
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        now_loss = loss.item()

        # print statistics
        running_loss += now_loss
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            if(min > running_loss / 2000):
                min = running_loss / 2000
                PATH = './cifar_net.pth'
                torch.save(net.state_dict(), PATH)
            running_loss = 0.0

print('Finished Training')


# Save the trained model
#PATH = './cifar_net.pth'
#torch.save(net.state_dict(), PATH)

# Load the pre_trained model
net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        ouputs = net(images)
        _, predicted = torch.max(ouputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))