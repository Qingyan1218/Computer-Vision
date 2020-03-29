#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from generate_train_test_list import crop_img,plot_face,read_file

# # 由于人脸识仅针对人脸,因此缩放、平移无意义，由于关键点基本为对称，所以水平翻转无意义
# # 由于最后是灰度图进行训练，因此变色等增广无意义
# # 此处采用旋转对图像进行增广，不涉及倒置的人脸
def rotate_bound(image, angle, scale):
    """旋转图像，并返回旋转后的图像和单应性矩阵
    @param:image:读入的图像数据
    @param:angle:旋转角度
    @param:scale:图像缩放比例"""
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵, 中心，角度，缩放
    M = cv2.getRotationMatrix2D((cx, cy), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH)), M

def trans_point(point,M):
    """单应性矩阵点乘坐标点矩阵，返回新的坐标
    @param:point:一个np.ndarray坐标点，nx2维，
    @param:M:单应性矩阵，2x3维"""
    trans_p=np.hstack((point,np.ones((point.shape[0],1))))
    new_point=np.dot(M,trans_p.T).T
    return new_point

def rotate_img(img,rect,pts,angle,scale=1.0):
    """返回旋转后的图像，边框及坐标点，均为np.ndarray
    @param:img:读入的原始图像数据
    @param:rect：图框,列表形式
    @param:pts：原始坐标点，列表形式"""
    pts=np.array(pts)
    new_img,M=rotate_bound(img,angle,scale)
    center = np.array([(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2])
    new_center=trans_point(center.reshape(1,2),M).reshape(2,)

    height = rect[3] - rect[1]
    width = rect[2] - rect[0]
    new_rect = [new_center[0] - width / 2, new_center[1] - height / 2,
                new_center[0] + width / 2, new_center[1] + height / 2]
    new_pts=trans_point(pts,M)
    return new_img,new_rect,new_pts

if __name__=='__main__':
    # # 产生增广图片
    file_path='.\\data\\III\\label.txt'
    file=open(file_path,'w')
    folder_list = ['.\\data\\I\\label.txt', '.\\data\\II\\label.txt']
    data=read_file(folder_list)
    for i in range(1000):
        row=random.choice(data)
        angle=np.random.randint(-90,90)
        img_name=row[0]
        img=cv2.imread(img_name)
        init_rect = row[1]
        pts=row[2]

        rect = [init_rect[0] / 1.2, init_rect[1] / 1.2, init_rect[2] * 1.2, init_rect[3] * 1.2]
        # plot_face(img,pts,rect)
        # plt.show()

        new_img, new_rect, new_pts = rotate_img(img,rect,pts,angle,1.0)
        # 每张图片必须保存一个新的名字，否则图片会覆盖，但是角度会不对
        cv2.imwrite(".\\data\\III\\new"+str(i)+img_name[-10:], new_img)
        file.write('new'+str(i)+img_name[-10:]+' ')
        for i in new_rect:
            file.write(str(i)+' ')
        for i in new_pts.flatten():
            file.write(str(i)+' ')
        file.write('\n')
    file.close()

    # 抽取一些图片进行验证
    folder_list = ['.\\data\\III\\label.txt']
    data=read_file(folder_list)
    for row in data[50:70]:
        img=cv2.imread(row[0])
        height=img.shape[0]
        width=img.shape[1]
        rect=row[1]
        pts=row[2]
        plot_face(img,pts,rect)
        plt.show()

