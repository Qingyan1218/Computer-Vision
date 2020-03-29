#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import cv2
import numpy as np
import matplotlib.pyplot as plt
from generate_train_test_list import read_file,crop_img

def get_iou(rect1, rect2):
    """计算交并比iou并返回
    @param:rect1和rect2是两个矩形对角坐标的列表
    """
    # rect: 0-4: x1, y1, x2, y2
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    #print('wleft: ', w_left, '  hleft: ', h_left, '    wright: ', w_right, '    h_right: ', h_right)

    box1_area = width1 * height1
    box2_area = width2 * height2
    #print('inner_area: ', inner_area, '   b1: ', box1_area, '   b2: ', box2_area)
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou

neg_gen_thre=100
random_border=112
negsample_ratio=0.3

def generate_random_crops(shape, rects, random_times):
    """生成随机的剪切，返回包含剪切框的列表
    @param:shape:图片的尺寸，hxw
    @param:rects:一个包括多个边框列表的列表，因为一张图片可能有几个人脸框
    @param:random_times:随机倍数"""
    neg_gen_cnt = 0
    img_h = shape[0]
    img_w = shape[1]
    rect_wmin = img_w   # 设定边框最小宽度初值
    rect_hmin = img_h   # 设定边框最小高度初值
    rect_wmax = 0
    rect_hmax = 0
    # num_rects是给定的边框数量
    num_rects = len(rects)
    for rect in rects:
        # 人脸框的宽度和高度
        w = rect[2] - rect[0] + 1
        h = rect[3] - rect[1] + 1
        # 如果边框的宽度小于最小宽度
        if w < rect_wmin:
            # 新的最小宽度等于w
            rect_wmin = w
        # 如果边框的宽度大于最大宽度
        if w > rect_wmax:
            # 新的最大宽度=w
            rect_wmax = w
        if h < rect_hmin:
            rect_hmin = h
        if h > rect_hmax:
            rect_hmax = h
    # 对于一个人脸框位于图片中间的图片，最后rect_wmin=w,rect_hmin=h,rect_wmax=0,rect_hmax=0
    random_rect_cnt = 0
    random_rects = []
    while random_rect_cnt < num_rects * random_times and neg_gen_cnt < neg_gen_thre:
        # 当random_rect_cnt< 人脸框数量x随机倍数 并且声称的数量小于阈值100
        neg_gen_cnt += 1
        # 如果图片高度-rect_hmax-训练时的边界高
        if img_h - rect_hmax - random_border > 0:
            # 则顶点从1到这个值中随机产生一个
            top = np.random.randint(0, img_h - rect_hmax - random_border)
        else:
            # 否则边界到顶
            top = 0
        # 宽度同理
        if img_w - rect_wmax - random_border > 0:
            left = np.random.randint(0, img_w - rect_wmax - random_border)
        else:
            left = 0
        # 随机从最小值和最大值中生成一个整数
        rect_wh = np.random.randint(min(rect_wmin, rect_hmin), max(rect_wmax, rect_hmax) + 1)
        rect_randw = np.random.randint(-3, 3)
        rect_randh = np.random.randint(-3, 3)
        right = left + rect_wh + rect_randw - 1
        bottom = top + rect_wh + rect_randh - 1

        good_cnt = 0
        for rect in rects:
            img_rect = [0, 0, img_w - 1, img_h - 1]
            rect_img_iou = get_iou(rect, img_rect)
            if rect_img_iou > negsample_ratio:
                random_rect_cnt += random_times
                break
            random_rect = [left, top, right, bottom]
            iou = get_iou(random_rect, rect)

            if iou < 0.3:
                # good thing
                good_cnt += 1
            else:
                # bad thing
                break

        if good_cnt == num_rects:
            # print('random rect: ', random_rect, '   rect: ', rect)
            _iou = get_iou(random_rect, rect)

            # print('iou: ', iou, '   check_iou: ', _iou)
            # print('\n')
            random_rect_cnt += 1
            random_rects.append(random_rect)
    return random_rects

if __name__=='__main__':
    folder_list = ['.\\data\\I\\label.txt', '.\\data\\II\\label.txt']
    path='.\\data\\IV\\label.txt'
    file=open(path,'w')
    data=read_file(folder_list)
    new_data={}
    for row in data:
        img_name=row[0]
        img = cv2.imread(img_name,0)
        size = img.shape
        rect = row[1]
        if img_name in new_data.keys():
            new_data[img_name][1].append(rect)
        else:
            new_data[img_name]=[None,[]]
            new_data[img_name][0]=size
            new_data[img_name].append(rect)

    print(len(new_data))
    for key,value in new_data.items():
        img_name=key
        shape=value[0]
        rects=value[1]
        new_rects=generate_random_crops(shape,rects,2)
        a=0
        if new_rects:
            img=cv2.imread(img_name,0)
            for r in new_rects:
                i_img=crop_img(img,r)
                cv2.imwrite(".\\data\\IV\\no" + str(a) + img_name[-10:], i_img)
                file.write("no" + str(a) + img_name[-10:]+' ')
                file.write(str(1)+' '+str(1)+' '+str(shape[0])+' '+str(shape[1])+' '+str(0)+'\n')
                a+=1

