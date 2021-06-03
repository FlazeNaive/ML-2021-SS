import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from L2Netmodel import L2Net

BATCH_SIZE = 512
EPOCHS = 25 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = L2Net(1, 10).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size = BATCH_SIZE, shuffle = True)

def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
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
torch.save(model, 'model.pth')
print("model saved")

quit()

BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = L2Net(1, 10).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size = BATCH_SIZE, shuffle = True)

def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if(batch_idx+1)%30 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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
    test(model, DEVICE, test_loader)
