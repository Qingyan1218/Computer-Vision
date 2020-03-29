#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu


import numpy as np
import cv2
import random
import os

# calculate means and std
train_txt_path = './train_label.csv'

# 挑选多少图片进行计算
CNum = 4572

img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    # random.shuffle(lines)

    # 第一行是标签，要略过
    for i in range(1,CNum):
        img_path = os.path.join(lines[i].rstrip().split(',')[1])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        if (i+1)%100==0:
            print(i)
        imgs = np.concatenate((imgs, img), axis=3)


imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    # 将第三维中的每一维（RGB中的一个）拉成一行
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
# BGR --> RGB
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize({},{})'.format(means, stdevs))
