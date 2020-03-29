#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os.path
import cv2
import random


def remove_invalid_image(data):
    """移除无效的图片文件，返回和data一样的列表
    @param：data：一个列表，记录图片位置，边框角点，关键点坐标"""
    images = []
    for line in data:
        name = line[0]
        if os.path.isfile(name):
            images.append(line)
    return images

def read_file(folder_list,read_ori_data=1):
    """读取label.txt中的数据，返回处理好的数据列表[文件名，边框列表，关键点列表]
    @param:folder_list:一个文件位置的列表
    @param:read_ori_data：如果是首次读取label，则为1，否则就是读取train.txt"""
    data = []
    for path in folder_list:
        lines=open(path).readlines()
        img_dir,_=os.path.split(path)
        for line in lines:
            single_file=[]
            line_data=line.strip().split()
            # 如果是读取原始文件，则需要补充文件目录，否则读取train.txt，无需补充目录
            if read_ori_data:
                single_file.append(os.path.join(img_dir,line_data[0]))
            else:
                single_file.append(line_data[0])
            single_file.append([float(line_data[i]) for i in range(1,5)])
            keypoints=line_data[5:]
            keypoints_list=[[float(keypoints[i]),float(keypoints[i+1])] for i in range(len(keypoints)) if i%2==0]
            single_file.append(keypoints_list)
            data.append(single_file)
    return data

def plot_face(img,pts,rect=None,color='r'):
    """绘制人脸图像，边框即关键点
    @param:img:读入的图像数据
    @param:pts:关键点坐标，列表形式，一对坐标是一个列表
    @param:rect:如果有rect，则绘制图框"""
    if len(img.shape)==3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    for point in pts:
        if point[0]<0:
            point[0]=0
        if point[1]<0:
            point[1]=0
        plt.scatter(point[0],point[1],c=color)
    if rect:
        H=rect[3]-rect[1]
        W=rect[2]-rect[0]
        currentAxis=plt.gca()
        rect=patches.Rectangle((rect[0],rect[1]),W,H,linewidth=1,edgecolor='g',facecolor='none')
        currentAxis.add_patch(rect)

def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):
    """将人脸框的范围扩大，返回关键点坐标及边长
    @param:x1~y2:图片关键点坐标
    @param:img_width:图片宽度
    @param:img_height:图片高度
    @param:ratio:边框扩大比例"""
    # usually ratio = 0.25
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)
    # 左上角的坐标减去填充宽度，右下角的坐标加上填充宽度
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height
    # 超出范围部分要裁剪
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2
    return roi_x1, roi_y1, roi_x2, roi_y2, \
           roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


def crop_img(img,crop_place):
    """裁剪图片，使用crop_place图片边框进行裁剪，返回新的图片
    @param:img:读入的图像数据
    @param:crop_place:需要裁减的边框列表"""
    if len(img.shape)==3:
        new_img=img[int(crop_place[1]):int(crop_place[3]),int(crop_place[0]):int(crop_place[2]),:]
    else:
        new_img = img[int(crop_place[1]):int(crop_place[3]), int(crop_place[0]):int(crop_place[2])]
    return new_img

def cut_img(img,rect,pts,ratio=0.25):
    """将图片的边框扩大，并且重新计算关键点相对于扩大的边框的坐标，返回新的图片，边框及坐标列表
    @param:img:读入的图像数据
    @param:rect:边框列表
    @param:pts:关键点坐标列表
    @param:ratio:边框扩大的比例"""
    height = img.shape[0]
    width = img.shape[1]
    new_rect = list(expand_roi(rect[0], rect[1], rect[2], rect[3], width, height, ratio))[0:4]
    new_pts = np.array(pts) - np.array([new_rect[0], new_rect[1]])
    new_img = crop_img(img,rect)
    return new_img,new_rect,new_pts

def save_file(data,file,train_ratio):
    """将数据写入train.txt和test.txt文件中
    @param:data:数据文件，由read_file函数生成
    @param:file:文件列表
    @param:train_ratio:训练数据的比例"""
    train_file=open(file[0],'w')
    test_file=open(file[1],'w')
    file_quantity=len(data)
    number=int(file_quantity*train_ratio)
    # 打乱数据集，因为III文件夹里的图片是旋转过的
    random.shuffle(data)
    for single_data in data[0:number]:
        train_file.write(single_data[0]+' ')
        for i in single_data[1]:
            train_file.write(str(i)+' ')
        # 增加一个数表示人脸
        train_file.write(str(1)+' ')
        for j in single_data[2]:
            train_file.write(str(j[0])+' '+str(j[1])+' ')
        train_file.write('\n')
    for single_data in data[number:]:
        test_file.write(single_data[0]+' ')
        for i in single_data[1]:
            test_file.write(str(i)+' ')
        # 增加一个数表示人脸
        test_file.write(str(1)+' ')
        for j in single_data[2]:
            test_file.write(str(j[0])+' '+str(j[1])+' ')
        test_file.write('\n')
    train_file.close()
    test_file.close()

def add_data_to_file(file_list,src_file,ratio):
    """将无人脸图片的信息写入train.txt和test.txt中
    @param:file_list:代表需要增加数据的train.txt和test.txt
    @param:src_file:无人脸图片的label.txt
    @param:ratio:需要切分到train和test中的比例"""
    data=open(src_file,'r').readlines()
    train_data=data[0:int(len(data)*ratio)]
    test_data=data[int(len(data)*ratio):]
    train_file=open(file_list[0],'a')
    for line in train_data:
        line='.\\data\\IV\\'+line
        train_file.write(line)
    train_file.close()
    test_file=open(file_list[1],'a')
    for line in test_data:
        line='.\\data\\IV\\'+line
        test_file.write(line)
    test_file.close()


def zoom(img,output_size,src_pts):
    """缩放图片后相应的修改关键点位置，返回新的图片，新的坐标
    @param:img:读入的图像数据
    @param:output:输出图像尺寸（WxH）
    @param:src_pts:原始关键点坐标，列表形式"""
    height=img.shape[0]
    width=img.shape[1]
    new_img = cv2.resize(img, output_size)
    height_ratio=output_size[1]/height
    width_ratio=output_size[0]/width
    pts=np.array(src_pts)
    new_pts_width = (pts[:, 0] * width_ratio).reshape(-1, 1)
    new_pts_height = (pts[:, 1] * height_ratio).reshape(-1, 1)
    new_pts = np.hstack((new_pts_width, new_pts_height))
    return new_img,new_pts

if __name__=='__main__':
    # label中总的信息数量是2783张，对于多于一张脸的图片，每张脸都有一条记录，增广1000张
    # img.shape=height,width,channel
    folder_list = ['.\\data\\I\\label.txt', '.\\data\\II\\label.txt', '.\\data\\III\\label.txt']
    data=remove_invalid_image(read_file(folder_list))
    new_data=[]
    for row in data:
        single_line=[]
        img=cv2.imread(row[0])
        height=img.shape[0]
        width=img.shape[1]
        rect=row[1]
        pts=row[2]
        # plot_face(img,pts,rect)
        # plt.show()
        new_img,new_rect,new_pts=cut_img(img,rect,pts)
        # plot_face(new_img,new_pts)
        # plt.show()
        single_line.append(row[0])
        single_line.append(new_rect)
        single_line.append(new_pts)
        new_data.append(single_line)

    train_ratio=0.8
    save_file(new_data,['train.txt','test.txt'],train_ratio)
    add_data_to_file(['train.txt','test.txt'],'.\\data\\IV\\label.txt',train_ratio)



