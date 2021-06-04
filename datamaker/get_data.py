import os
import cv2
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')
print("=============start=============")

_PATH = "/home/FlazeH/Desktop/myL2Net/data/hpatches-release" #/i_ajuntament"

count = 0
for FOLDER in os.listdir(_PATH):
    PATH = os.path.join(_PATH, FOLDER)
    # count = count + 1
    # if count > 10:
    #     break

    imglist = []
    for s in os.listdir(PATH):
        if "ref.png" in s:
            # print(s)
            tmp_img = cv2.cvtColor(cv2.imread(os.path.join(PATH,s)),cv2.COLOR_BGR2GRAY)
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=32/65.0, fy=32/65.0, interpolation=cv2.INTER_NEAREST)
            imglist.append(tmp_img)

    for s in os.listdir(PATH):
        if ("png" in s) and ("e" in s) and not("ref" in s):
            # print(s)
            tmp_img = cv2.cvtColor(cv2.imread(os.path.join(PATH,s)),cv2.COLOR_BGR2GRAY)
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=32/65.0, fy=32/65.0, interpolation=cv2.INTER_NEAREST)
            imglist.append(tmp_img)

    for s in os.listdir(PATH):
        if ("png" in s) and ("h" in s):
            # print(s)
            tmp_img = cv2.cvtColor(cv2.imread(os.path.join(PATH,s)),cv2.COLOR_BGR2GRAY)
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=32/65.0, fy=32/65.0, interpolation=cv2.INTER_NEAREST)
            imglist.append(tmp_img)

    for s in os.listdir(PATH):
        if ("png" in s) and ("t" in s):
            # print(s)
            tmp_img = cv2.cvtColor(cv2.imread(os.path.join(PATH,s)),cv2.COLOR_BGR2GRAY)
            tmp_img = cv2.resize(tmp_img, (0, 0), fx=32/65.0, fy=32/65.0, interpolation=cv2.INTER_NEAREST)
            imglist.append(tmp_img)

    print("=================end============")
    print("find " + str(len(imglist)) )
    print("=================processing============")

    [hh, ww] = np.shape(imglist[0])

    # tmp_to_show = np.zeros([0, 32*16])
    i_ajuntament = []

    for i in range(round(hh/ww)):
        tmp = []
        for j in range(16):
            tt = imglist[j][i*32:(i+1)*32, :]
            tmp.append(tt)
        i_ajuntament.append(tmp)
        # tmp_to_show = np.concatenate((tmp_to_show, tmp), axis = 0)

    i_ajuntament = np.array(i_ajuntament)
    # print(i_ajuntament)
    print("saving " + FOLDER + ".npy")
    np.save("data/"+FOLDER+".npy", i_ajuntament)
    # sxxxx = input()
    # plt.imshow(tmp_to_show, cmap=plt.cm.gray)
    # plt.show()

'''
data = np.load("i_ajuntament.npy")

for view in data:
    plt.imshow(view, cmap=plt.cm.gray)
    plt.show()

'''