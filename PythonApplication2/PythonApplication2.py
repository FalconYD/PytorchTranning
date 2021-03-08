import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

import cv2 as cv
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), #텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=False, # False를 지정하면 테스트 데이터로 다운로드
                          transform=transforms.ToTensor(), #텐서로 변환
                          download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #   Conv    -> (?, 28, 28, 32)
        #   Pool    -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

#model = CNN().to(device)
#
##criterion = torch.nn.CrossEntropyLoss.to(device)
#criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#total_batch = len(data_loader)
#print('총 배치의 수 : {}'.format(total_batch))
#
#for epoch in range(training_epochs):
#    avg_cost = 0
#    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블
#        # image is already size of (28x28), no reshape
#        # label is not one-hot encoded
#        X = X.to(device)
#        Y = Y.to(device)
#
#        optimizer.zero_grad()
#        hypothesis = model(X)
#        cost = criterion(hypothesis, Y)
#        cost.backward()
#        optimizer.step()
#
#        avg_cost += cost / total_batch
#
#    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# Save the trained model
PATH = './cifar_net.pth'
#torch.save(model.state_dict(), PATH)
import torchvision.models as models;
model = models.resnet50(pretrained=True).to(device);

model = CNN().to(device)
model.load_state_dict(torch.load(PATH))

# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    print(X_test.shape)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction,1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:',accuracy.item())


#-----------------------------------------------------
from torch.utils.data import Dataset

class MyDataset(Dataset):
    
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)
        
        return  sample

    def __len__(self):
        return self.len

class ToTensor:

    def __call__(self, sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2,0,1)
        return inputs, torch.LongTensor(labels)

class LinearTensor:

    def __init__(self, slope=1, bias=0):
        self.slope = slope
        self.bias = bias

    def __call__(self, sample):
        inputs, labels = sample
        inputs = self.slope*inputs + self.bias

        return inputs, labels

class CvtColor:
    def __call__(self, Img):
        inputs = Img
        inputs = torch.FloatTensor(inputs)
        return inputs

width = 28
height = 28

file = 'img0.jpg'
#file = 'IMG_2.png'
#Loading the file
#img2 = cv.imread(file, cv.IMREAD_COLOR)
img2 = cv.imread(file, cv.IMREAD_GRAYSCALE)

#Format for the Mul:0 Tensor
img2 = cv.resize(img2, dsize=(width,height), interpolation = cv.INTER_CUBIC)

cv.imshow('IMG',img2)
cv.waitKey(0)

img2 = cv.bitwise_not(img2)
cv.imshow('IMG',img2)
cv.waitKey(0)
cv.destroyAllWindows()

#Numpy array
np_image_data = np.resize(img2, (28,28, 1))

#np_image_data = np.asarray(img2)

img = torch.FloatTensor(np_image_data)
img.permute(2,0,1)
print(img.shape)

dset = MyDataset(img, '2')
# 메모리 공유
#img = torch.from_numpy(np_image_data)
#-----------------------------------------------------

with torch.no_grad():
    X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    test = dset.x_data.view(len(dset), 1, 28, 28).float().to(device)

    prediction = model(test)
    #prediction = model(img)
    result = torch.argmax(prediction,1)
    print('Accuracy:', result)
    #correct_prediction = result == Y_test
    #accuracy = correct_prediction.float().mean()
    #print('Accuracy:',accuracy.item())