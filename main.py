import torch
from torch import tensor
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import os
import numpy as np
import random

from L2Netmodel import L2Net
from make_dataset import HPatchesDataset

EPOCHS = 80
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = L2Net(1, 128).to(DEVICE)
optimizer = optim.Adadelta(model.parameters(), lr=0.01)

def E1(dift_codes):
    Y1 = dift_codes[0] 
    Y2 = dift_codes[1] 
    
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

def E2(dift_codes):
    # return 0

    Y1 = dift_codes[0] 
    Y2 = dift_codes[1] 
    
    #E2
    Y1_mean = torch.mean(Y1, dim = 1)
    Y2_mean = torch.mean(Y2, dim = 1)

    Y1_tmp = (Y1.t()- Y1_mean).t()
    Y2_tmp = (Y2.t()- Y2_mean).t()

    Y1_norm = torch.norm(Y1_tmp, p = 2, dim = 1)
    Y2_norm = torch.norm(Y2_tmp, p = 2, dim = 1)
    # print(Y1_norm)

    Y1_up = Y1_tmp.mm(Y1_tmp.t())
    Y2_up = Y2_tmp.mm(Y2_tmp.t())

    Y1_up = (Y1_up.t() / Y1_norm).t()
    # print(Y1_up)
    Y1_up = Y1_up / Y1_norm
    # print(Y1_up)
    # print("UUU")
    Y2_up = Y2_up / Y2_norm
    # print(Y2_up)
    Y2_up = (Y2_up.t() / Y2_norm).t()
    # print(Y2_up)

    r_ij_1 = Y1_up.mul(Y1_up) 
    r_ij_1 = torch.sum(r_ij_1) - torch.sum(torch.diag(r_ij_1))
    r_ij_2 = Y2_up.mul(Y2_up) 
    r_ij_2 = torch.sum(r_ij_2) - torch.sum(torch.diag(r_ij_2))

    tmp_E2 = 0.5*(r_ij_1 + r_ij_2)
    return tmp_E2

def E3(dift_codes):
    Y1 = dift_codes[0] 
    Y2 = dift_codes[1] 
    tmp_E3 = 0
    return tmp_E3

train_data = HPatchesDataset("/home/FlazeH/Desktop/myL2Net/datamaker/train")
test_data = HPatchesDataset("/home/FlazeH/Desktop/myL2Net/datamaker/test")
# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=False)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False)

def train(model, device, optimizer, epoch):
        model.train()
        for batch_base in range(0, train_data.__len__() - 16 * BATCH_SIZE, 16 * BATCH_SIZE):
            X_1 = torch.zeros([0, 1, 32, 32])
            target_1 = [] # torch.zeros([])
            X_2 = torch.zeros([0, 1, 32, 32])
            target_2 = [] # torch.zeros([])
            for y_base in range(batch_base, batch_base + 16 * BATCH_SIZE, 16):
                bias_1 = random.randint(0, 15)
                bias_2 = random.randint(0, 15)
                while(bias_2 == bias_1):
                    bias_2 = random.randint(0, 15)
                tmp_data1 = train_data[y_base + bias_1][0]
                tmp_data2 = train_data[y_base + bias_2][0]
                tmp_data1 = torch.reshape(tmp_data1, (1, 1, 32, 32))
                tmp_data2 = torch.reshape(tmp_data2, (1, 1, 32, 32))
                tmp_target1 = train_data[y_base + bias_1][1]
                tmp_target2 = train_data[y_base + bias_2][1]
                X_1 = torch.cat((X_1, tmp_data1), 0)
                X_2 = torch.cat((X_2, tmp_data2), 0)
                target_1.append(tmp_target1)
                target_2.append(tmp_target2)
                # target_1 = torch.cat((target_1, tmp_target1), 0)
                # target_2 = torch.cat((target_2, tmp_target2), 0)
            
            data = torch.cat((X_1, X_2), 0).to(device)
            target = torch.cat((torch.tensor(target_1), torch.tensor(target_2)), 0).to(device)
            data, target = Variable(data), Variable(target)

            '''
        for batch_idx, (data, target) in enumerate(train_loader):
            print(type(data))
            print(data.shape)
            print(type(target))
            print(target.shape)
            # quit()

            data, target = data.to(device), target.to(device)
            '''
            
            optimizer.zero_grad()
            output = model(data)
            # L2_dis = torch.norm(output[:, None] - output, dim=2, p=2)
            # print(L2_dis[0])
            # print(output[0])
            # print(norm(output[0]))
            # print(output.shape)
            # quit()
            out_Y = [output[0 : BATCH_SIZE :], output[BATCH_SIZE : BATCH_SIZE*2, :]]
            loss = E1(out_Y) + E2(out_Y)
            loss.backward()
            optimizer.step()
            if(batch_base/(BATCH_SIZE * 16)+1)%10 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_base/(BATCH_SIZE * 16) * len(data), train_data.__len__(),
                    100. * batch_base/(BATCH_SIZE*16) / (train_data.__len__() / (BATCH_SIZE * 16)), loss.item()))

test_corrects = []
test_losses = []
def test(epoch_num, model, device):
    model.eval()
    test_loss = 0
    test_num = 0
    correct = 0
    total_check = 0
    with torch.no_grad():
        for batch_base in range(0, test_data.__len__() - 16 * BATCH_SIZE, 16 * BATCH_SIZE):
            X_1 = torch.zeros([0, 1, 32, 32])
            target_1 = [] # torch.zeros([])
            X_2 = torch.zeros([0, 1, 32, 32])
            target_2 = [] # torch.zeros([])
            for y_base in range(batch_base, batch_base + 16 * BATCH_SIZE, 16):
                bias_1 = random.randint(0, 15)
                bias_2 = random.randint(0, 15)
                while(bias_2 == bias_1):
                    bias_2 = random.randint(0, 15)
                tmp_data1 = test_data[y_base + bias_1][0]
                tmp_data2 = test_data[y_base + bias_2][0]
                tmp_data1 = torch.reshape(tmp_data1, (1, 1, 32, 32))
                tmp_data2 = torch.reshape(tmp_data2, (1, 1, 32, 32))
                tmp_target1 = test_data[y_base + bias_1][1]
                tmp_target2 = test_data[y_base + bias_2][1]
                X_1 = torch.cat((X_1, tmp_data1), 0)
                X_2 = torch.cat((X_2, tmp_data2), 0)
                target_1.append(tmp_target1)
                target_2.append(tmp_target2)
                # target_1 = torch.cat((target_1, tmp_target1), 0)
                # target_2 = torch.cat((target_2, tmp_target2), 0)
            
            data = torch.cat((X_1, X_2), 0).to(device)
            target = torch.cat((torch.tensor(target_1), torch.tensor(target_2)), 0).to(device)
            data, target = Variable(data), Variable(target)

            output = model(data)
            out_Y = [output[0 : BATCH_SIZE, :], output[BATCH_SIZE : 2 * BATCH_SIZE, :]]
            test_loss += E1(out_Y) + E2(out_Y)
            test_num = test_num + 1

            L2_dis = torch.norm(output[:, None] - output, dim=2, p=2)
            L2_dis = L2_dis.cpu()
            answer = target.cpu()
            for i in range(BATCH_SIZE):
                total_check = total_check + 1
                nrst = np.argmin(L2_dis[i, BATCH_SIZE:BATCH_SIZE*2])
                # print(i)
                # print(nrst)
                # print("=====")
                if nrst == i :                        
                    correct = correct + 1

    print("test_loss = ", test_loss / test_num)
    print("correct = ", correct/ total_check)
    test_corrects.append(correct/total_check)
    test_losses.append(test_loss / test_num)

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, optimizer, epoch)
    test(epoch, model, DEVICE)

print("train end\n\nSaving testing data...")
# np.save("E1+E2_test_loss", test_losses)
# np.save("E1+E2_test_corr", test_corrects)
np.save("E1_test_loss", test_losses)
np.save("E1_test_corr", test_corrects)
print("OK")
print("Saving model...")
# torch.save(model.state_dict(), "E1+E2_model.pth")
torch.save(model.state_dict(), "E1_model.pth")
print("OK")
