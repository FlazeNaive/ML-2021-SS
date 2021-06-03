import os
import cv2
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

'''
for view in data:
    iii = np.zeros([32, 0])
    for img in view:
        iii = np.concatenate((iii, img), 1)
    plt.imshow(iii, cmap=plt.cm.gray)
    plt.show()
'''

class HPatchesDataset(Dataset):
    def __init__(self, PATH, transform = None):
        super().__init__()
        self.PATH= PATH 
        self.data = []
        self.label = []
        self.count = 0
        self.transform = transform
        for s in os.listdir(self.PATH):
            if "npy" in s:
                tmp_p = os.path.join(self.PATH, s)
                print(tmp_p)
                datafile = np.load(tmp_p)
                for (i, view) in enumerate(datafile):
                    for (j, img) in enumerate(view):
                        img_ = torch.from_numpy(img.reshape(1, 32, 32)/255).float()
                        # print(img_)
                        label_ = torch.tensor(self.count)
                        self.data.append(img_)
                        self.label.append(label_)
                        # print(type(img_))
                        # print(type(label_))
                    self.count = self.count + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        sample = (img, label)#根据图片和标签创建字典
        
        if self.transform:
            sample = self.transform(sample)#对样本进行变换
        return sample #返回该样本

'''
data = HPatchesDataset("/home/FlazeH/Desktop/myL2Net/datamaker/")
dataloader = DataLoader(data, batch_size=128,shuffle=True)#使用DataLoader加载数据
for i_batch,batch_data in enumerate(dataloader):
    print(i_batch)#打印batch编号
    print("image size")
    print(batch_data['image'].size())#打印该batch里面图片的大小
    print(batch_data['label'])#打印该batch里面图片的标签
'''