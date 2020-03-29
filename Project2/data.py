import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
from generate_train_test_list import crop_img,zoom,plot_face,cut_img
from img_augmentation import rotate_img

train_boarder = 112

def channel_norm(img):
    """将图片像素值标准化，返回标准化后的数据
    @param:img:读入的图像数据，np.ndarray形式"""
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels

# def reverse_channel_norm(data,mean,std):
#     return std*data+mean


def parse_line(line):
    """解析一行数据，将train或text.txt的一行变成3个部分，
    可以使用generate_train_test_list_myself中的read_file,
    该函数返回data数据，每一个元素就是四个部分的组合
    @param:line:文件中的一行信息"""
    line_parts = line.strip().split()
    img_name = line_parts[0]

    rect = list(map(int, list(map(float, line_parts[1:5]))))
    label = int(line_parts[5])
    if label ==1:
        landmarks = list(map(float, line_parts[6: len(line_parts)]))
    else:
        # 如果不是人脸，让其回归到0
        landmarks = np.zeros(42,)
    return img_name, rect, label,landmarks

def gen_augmentation_lines(line,angle,scale):
    """该函数用来增广数据行，和data中的I和II文件夹下的一致，
    包括原始的图片名，原始的图框和坐标点
    @param:line:一行数据
    @param:angle:旋转角度
    @param:scale:缩放比例"""
    img_name,rect,landmarks=parse_line(line)
    img=cv2.imread(img_name,0)
    # 获取旋转后的图像，边框及关键点，相当于一个原始的图像信息
    pts=[[landmarks[i], landmarks[i + 1]] \
                          for i in range(len(landmarks)) if i % 2 == 0]
    new_img, new_rect, new_pts = rotate_img(img,rect,pts,angle,scale)
    move_img,move_rect,move_pts=cut_img(new_img,new_rect,new_pts,ratio=0.4)
    new_line_info=[img_name]+[str(i) for i in move_rect]+[str(j) for j in move_pts.flatten()]
    new_line=' '.join(new_line_info)
    return new_line


# 该类用来标准化
class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        # sample是个字典，包括了image和landmarks两个键
        image, landmarks ,label= sample['image'], sample['landmarks'],sample['label']
        # 图片缩放为训练尺寸，关键点已经缩放好
        # image_resize = np.asarray(
        #                     image.resize((train_boarder, train_boarder), Image.BILINEAR),
        #                     dtype=np.float32)       # Image.ANTIALIAS)

        image_resize=cv2.resize(image, (train_boarder,train_boarder))
        # 标准化，由于图片尺寸不变，而坐标是绝对位置信息，因此不能标准化
        image= channel_norm(image_resize)
        # 由于绝对位置数值太大，损失函数马上就爆表了，所以landmarks最好缩小，

        return {'image': image,
                'landmarks': landmarks,
                'label':label
                }

# 该类将数据变成张量
class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks ,label= sample['image'], sample['landmarks'],sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        # 增加维度，符合训练要求

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label,axis=0)
        # 后面计算都是基于float的，默认是double的

        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'landmarks': torch.from_numpy(landmarks).type(torch.FloatTensor),
                'label':torch.from_numpy(label).type(torch.LongTensor)}

# 该类用来处理数据
class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase,transform=None):
        '''
        @param:src_lines: src_lines, read from train.txt or test.txt
        @param:train: whether we are training or not
        @param:transform: data transform
        '''
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, label, landmarks = parse_line(self.lines[idx])
        # image
        # 用PIL的Image函数读取图片并转成灰度图像
        # img = Image.open(img_name).convert('L')
        img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
        # 对图像进行裁剪
        img_crop = crop_img(img,rect)
        # 坐标已经对应于裁剪后的图片
        if label==1:
            landmarks = np.array(landmarks).astype(np.float32)

            keypoints_list = [[landmarks[i], landmarks[i + 1]] \
                              for i in range(len(landmarks)) if i % 2 == 0]
		
            _,landmarks=zoom(img_crop,(train_boarder,train_boarder),keypoints_list)
            landmarks=np.array(landmarks).flatten()
        sample = {'image': img_crop, 'landmarks': landmarks,'label':label}
        sample = self.transform(sample)
        return sample



def load_data(phase):
    """根据阶段读取相应的文件
    @param:phase:阶段"""
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()


    # 根据阶段对数据进行预处理
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, transform=tsfm)
    return data_set


def get_train_test_set():
    """获取训练和验证数据"""
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('train')
    # path = '.\\result_img\\'
    # for i in range(len(train_set)-20,len(train_set)):
    for i in range(1,20):
        sample = train_set[i]
        img = sample['image'].numpy().reshape(train_boarder,train_boarder)
        landmarks = sample['landmarks']
        label=sample['label']
        if label==1:
            keypoints_list = [[landmarks[i], landmarks[i + 1]] \
                          for i in range(len(landmarks)) if i % 2 == 0]
            # # 画出人脸以及对应的关键点
            plot_face(img,keypoints_list)
            plt.show()
            # # 保存图片供观察
            # plt.savefig(path+str(i)+'.jpg')
            # plt.close()









