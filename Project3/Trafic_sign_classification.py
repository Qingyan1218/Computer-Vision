#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
import random
import torch.optim as optim
from PIL import Image
import copy
import time

ROOT_DIR = '.\\traffic-sign'
TRAIN_DIR = '\\train'
VAL_DIR = '.\\test'
TRAIN_ANNO = 'train_label.csv'
VAL_ANNO = 'test_label.csv'
RESIZE_SIZE=112
BATCH_SIZE=16

output_dimension=62
learn_rate = 0.001
num_epoches = 50
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
img_mean=np.array([0.40086165, 0.37575805, 0.38072258])
img_std=np.array([0.2707049, 0.2707329, 0.28101894])

class MyDataset():

    def __init__(self, root_dir, annotations_file, transform=None):

        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.transform = transform

        if not os.path.isfile(self.annotations_file):
            print(self.annotations_file + ' does not exist!')
        self.file_info = pd.read_csv(annotations_file, index_col=0)
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.file_info['image_location'][idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        img = Image.open(image_path).convert('RGB')
        labels = int(self.file_info.iloc[idx]['class_id'])

        sample = {'image': img, 'label': labels}
        if self.transform:
            sample['image'] = self.transform(img)
        return sample


# 读入的图片无需除以255，ToTensor会自动将像素值归一化，所以Normalize要在ToTensor之后，
train_transforms = transforms.Compose([transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(45),
                                       transforms.ToTensor(),
                                       transforms.Normalize(img_mean,img_std)])

val_transforms = transforms.Compose([transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(img_mean,img_std)])

def visualize_dataset():
    idx = random.randint(0, len(train_dataset))
    sample = train_dataloader.dataset[idx]
    # sample是个tensor，形状是3x112x112，变成numpy后要转换通道，不要reshape
    img = sample['image'].numpy().transpose(1,2,0)
    img=img*img_std+img_mean
    plt.imshow((img*255).astype('uint8'))
    plt.show()

# 定义网络
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # # 搭建网络，卷册层和全连接层分开表示，层内不同的模块分开表示
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=0),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),

            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),

            torch.nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=0),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(24),
            torch.nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=0),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(24),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),

            torch.nn.Conv2d(24, 40, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(40),
            torch.nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(80),
        )

        self.FullConnection = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 80, 128),
            torch.nn.PReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 128),
            torch.nn.PReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, output_dimension)

        )

    def forward(self, x):
        x=self.Conv(x)
        x=x.view(-1, 4 * 4 * 80)
        x=self.FullConnection(x)
        return x


def train_model(model, criterion, optimizer, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_label = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    time_now=time.strftime('%Y_%m_%d_%H_%M_%S')
    file = open(time_now+'.txt','w')
    file.write(model.__class__.__name__ + '_'+ criterion.__class__.__name__ + '_' \
                              + optimizer.__class__.__name__ + '_' + str(learn_rate)+'\n')
    file.write(time_now+'\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)
        file.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        file.write('-*' * 10+'\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_labels = 0

            for idx,data in enumerate(data_loaders[phase]):
                #print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_true = data['label'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x = model(inputs)
                    x = x.view(-1,62)

                    # torch.max返回最大的那值和索引
                    _, preds = torch.max(x, 1)

                    loss = criterion(x, labels_true)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                corrects_labels += torch.sum(preds== labels_true)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_label = corrects_labels.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_label

            Accuracy_list_label[phase].append(100 * epoch_acc_label)
            print('{} Loss: {:.4f}  Acc_label: {:.2%}'.format(phase, epoch_loss,epoch_acc_label))
            file.write('{} Loss: {:.4f}  Acc_label: {:.2%}\n'.format(phase, epoch_loss,epoch_acc_label))

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc_label
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val label Acc: {:.2%}'.format(best_acc))
                file.write('Best val label Acc: {:.2%}\n'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model_'+time_now+'.pt')
    print('Best val label Acc: {:.2%}'.format(best_acc))
    file.write('Best val label Acc: {:.2%}\n'.format(best_acc))
    file.close()
    return model, Loss_list,Accuracy_list_label

def plot_save_curve(result_dict,num_epoches,label):
    x = range(0, num_epoches)
    y1 = result_dict["val"]
    y2 = result_dict["train"]
    plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
    plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
    plt.legend()
    plt.title('train and val loss vs. epoches')
    plt.ylabel(label)
    plt.show()
    # plt.savefig('train and val '+label+' vs epoches.jpg')
    # plt.close('all') # 关闭图 0


def visualize_model(model,data):
    model.eval()
    total=len(data)
    print(total)
    correct_label=0
    with torch.no_grad():
        for i, data in enumerate(data):
            inputs = data['image']
            labels_true = data['label'].to(device)

            x_label = model(inputs.to(device))
            # x_label = x_label.view(-1,62)
            _, preds_label = torch.max(x_label, 1)
            # 统计正确的个数，并输出分类错误的图片
            if labels_true==preds_label:
                correct_label+=1
            else:
                img=inputs.squeeze().numpy().transpose(1,2,0)
                img=img*img_std+img_mean
                plt.imshow((img*255).astype('uint8'))
                plt.title('predicted label: {}\n ground-truth label:{}'.format(preds_label,labels_true))
                plt.show()
    acc=correct_label/total
    print(f'accuracy: {acc*100:0.2f}%')


def load_model(model_file,src_model):
    model=src_model()
    model.load_state_dict(model_file)
    return model


if __name__=='__main__':
    train_dataset = MyDataset(root_dir=ROOT_DIR + TRAIN_DIR,
                              annotations_file=TRAIN_ANNO,
                              transform=train_transforms)

    test_dataset = MyDataset(root_dir=ROOT_DIR + VAL_DIR,
                             annotations_file=VAL_ANNO,
                             transform=val_transforms)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset)
    data_loaders = {'train': train_dataloader, 'val': test_dataloader}

    # visualize_dataset()

    # # 以下是一些经典模型，由于网络较大，训练费时，因此此处不用
    # my_vggnet = torchvision.models.vgg16(pretrained=False)
    # input_dimension = my_vggnet.classifier[6].in_features
    # my_vggnet.classifier[6] = torch.nn.Linear(input_dimension,output_dimension)
    # print(my_vggnet)

    # my_alexnet = torchvision.models.alexnet(pretrained=False)
    # input_dimension = my_alexnet.classifier[6].in_features
    # my_alexnet.classifier[6] = torch.nn.Linear(input_dimension,output_dimension)
    # print(my_alexnet)

    # my_resnet = torchvision.models.resnet(pretrained=False)
    # input_dimension = my_resnet.fc.in_features
    # my_resnet.fc = torch.nn.Linear(input_dimension,output_dimension)
    # print(my_resnet)

    # 定义超参数
    model = Model().to(device)
    # model = my_alexnet.to(device)
    # model = my_vggnet.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(),lr=learn_rate,momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    # model, Loss_list, Accuracy_list_label = train_model(model, criterion,
    #                                          optimizer,  num_epochs=num_epoches)
    #
    # plot_save_curve(Loss_list,num_epoches,'loss')
    # plot_save_curve(Accuracy_list_label,num_epoches,'accuracy')
    #
    #
    # visualize_model(model,data_loaders['val'])
    model.load_state_dict(torch.load('best_model_2020_03_29_13_05_51.pt'))
    visualize_model(model,data_loaders['val'])
