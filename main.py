import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import os
import numpy as np
from L2Netmodel import L2Net
from make_dataset import HPatchesDataset

EPOCHS = 10
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = L2Net(1, 128).to(DEVICE)
optimizer = optim.Adadelta(model.parameters(), lr=0.01)

def E1(dift_codes):
    Y1 = dift_codes[0] #[batch,diftcode_len]
    Y2 = dift_codes[1] #[batch,diftcode_len]
    
    #E1
    eps = 1e-6
    Y1_tmp = torch.unsqueeze(Y1,dim=0)
    Y2_tmp = torch.unsqueeze(Y2,dim=1)
    D_sub = Y1_tmp - Y2_tmp #(batch,batch,diftcode_len)
    D = torch.sqrt(torch.sum(D_sub*D_sub,dim=2)+1e-6)

    # D_mat_mul = torch.matmul(Y1,Y2.T)#(batch,batch)
    # D = torch.sqrt(2.0*(1.0-D_mat_mul+1e-6))#[batch,batch]               
    
    D_exp = torch.exp(2.0-D)
    D_ii = torch.unsqueeze(torch.diag(D_exp),dim=1)#[batch,1]
    #"compute_col_loss"
    D_col_sum = torch.sum(D_exp.T,dim=1,keepdim=True)#[batch,1]
    s_ii_c = D_ii / (eps+D_col_sum)
    #"compute_row_loss"
    D_row_sum = torch.sum(D_exp,dim=1,keepdim=True)#[batch,1]
    s_ii_r = D_ii / (eps+D_row_sum)
    
    tmp_E1 = -0.5*(torch.sum(torch.log(s_ii_c))+torch.sum(torch.log(s_ii_r)))
    return tmp_E1

train_data = HPatchesDataset("/home/FlazeH/Desktop/myL2Net/datamaker/train")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)#使用DataLoader加载数据
test_data = HPatchesDataset("/home/FlazeH/Desktop/myL2Net/datamaker/test")
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True)#使用DataLoader加载数据


def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # print(type(data))
            # print(type(target))

            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            
            '''
            image_array,_=train_dataset[batch_idx]
            image_array=image_array.reshape(28,28)
            plt.imshow(image_array)
            plt.show()
            '''

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            print(output)
            print(target)
            print("QAQ")
            # loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if(batch_idx+1)%30 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(epoch_num, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(epoch, model, DEVICE, test_loader)

print("train end")