from make_dataset import HPatchesDataset

import os
import cv2
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

data = HPatchesDataset("/home/FlazeH/Desktop/myL2Net/datamaker/dataloader_test/")
dataloader = DataLoader(data, batch_size=16,shuffle=False)#使用DataLoader加载数据
for i_batch,batch_data in enumerate(dataloader):
    print(i_batch)#打印batch编号
    plt.switch_backend('agg')
    tmp_to_show = batch_data[0][0, 0, :, :]
    for i in range(1, 16):
        tmp_to_show = np.concatenate((tmp_to_show, batch_data[0][i, 0, :, :]), axis=1)
    print(batch_data[1])
    plt.imshow(tmp_to_show, cmap=plt.cm.gray)
    plt.savefig('test_dataset.jpg', bbox_inches='tight')
    '''
    print("image size")
    print(batch_data[0].size())#打印该batch里面图片的大小
    print(batch_data[1])#打印该batch里面图片的标签
    '''
    xxx = input("done")