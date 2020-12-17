#import torch

#cpu_tensor = torch.zeros(2,3)
#device=torch.device("cuda:0")
#gpu_tensor = cpu_tensor.to(device)
#print(gpu_tensor)

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

import torch
import torch.nn as nn           #신경망 모델 중 Linear 함수
import torch.optim as optim     #경사하강법 알고리즘 
import torch.nn.init as init    #텐서 초기값을 주기위한 필요 함수


num_data=1000 #데이터 수
num_epoch=500 #선형회귀반복횟수

x = init.uniform_(torch.Tensor(num_data,1),-10,10) #-10, 10의 균등한 데이터
noise=init.normal_(torch.FloatTensor(num_data,1),std=1) #가우시안 노이즈

y=2*x+3 #일반 함수 (y=2x+3)
y_noise=2*(x+noise)+3 #노이즈 추가된 함수

#선형 회귀 모델
model=nn.Linear(1,1) #들어오는 특성, 결과로 나오는 특성의 수, 편차 사용 여부
loss_func=nn.L1Loss() #L1손실 차이의 절댓값의 평균.

#-------------------------------------