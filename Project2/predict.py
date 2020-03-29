from __future__ import print_function

import torch
import os
import cv2
import matplotlib.pyplot as plt

from generate_train_test_list import plot_face
from data import channel_norm,train_boarder

# 自己编写的预测函数，从文件夹中读取图片文件并依次预测
def my_predict(img_path,model):
    """使用模型预测图片中人脸位置，打印人脸及关键点
    @param:img_path:需要预测的图片位置
    @param:model:预测使用的模型"""
    img_list=os.listdir(img_path)
    for imgname in img_list:
        img_full_path=os.path.join(img_path,imgname)
        img=cv2.imread(img_full_path,0)
        img=cv2.resize(img,(train_boarder,train_boarder))
        # 标准化
        norm_img = channel_norm(img).reshape(1, 1, img.shape[0], img.shape[1])
        # 转变成torch张量
        data = torch.from_numpy(norm_img).type(torch.FloatTensor)
        with torch.no_grad():
            label_array,landmarks_array = model(data)
            _,label=torch.max(label_array,1)
            label=label.numpy()[0]
            landmarks=landmarks_array.data.numpy()[0]
            if label==1:
                pts = [[landmarks[i], landmarks[i + 1]] \
                    for i in range(len(landmarks)) if i % 2 == 0]
                plot_face(img, pts)
                plt.show()
            else:
                plt.imshow(img)
                plt.title('this picture has no face in it')
                plt.show()


# 此部分代码针对stage 1中的predict。 是其配套参考代码
# 对于stage3， 唯一的不同在于，需要接收除了pts以外，还有：label与分类loss。
def predict(args, trained_model, model, valid_loader):
    """使用模型预测图片中人脸位置，打印人脸及关键点
    @param:args:包含参数的args类
    @param:train_model:保存的训练模型
    @param:model:网络模型
    @param:valid_loader:验证用的数据集"""
    # 加载模型文件
    model.load_state_dict(torch.load(os.path.join(args.save_directory, trained_model)))   # , strict=False
    # 变成预测模式
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            # 图像
            img = batch['image']
            # 关键点坐标
            landmark = batch['landmarks']
            print('i: ', i)
            # 计算模型输出
            label_array,output_pts = model(img)
            _,label=torch.max(label_array,1)
            label=label.numpy()[0]
            # 变成numpy数组
            if label==1:
                outputs = output_pts.numpy()[0]
                print('outputs: ', outputs)
                # 匹配坐标点
                x = list(map(int, outputs[0: len(outputs): 2]))
                y = list(map(int, outputs[1: len(outputs): 2]))
                # landmarks_generated = list(zip(x, y))
                landmarks_generated = [list(i) for i in list(zip(x, y))]

                # 实际坐标点
                landmark = landmark.numpy()[0]
                x = list(map(int, landmark[0: len(landmark): 2]))
                y = list(map(int, landmark[1: len(landmark): 2]))
                # landmarks_truth = list(zip(x, y))
                landmarks_truth = [list(i) for i in list(zip(x, y))]
                # 转换图片
                img = img.numpy()[0].transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                plt.imshow(img)
                # 红点表示自动生成的，蓝点表示标记的
                plot_face(img,landmarks_generated,color='r')
                plot_face(img,landmarks_truth,color='b')
                plt.show()
            else:
                img = img.numpy()[0].transpose(1, 2, 0)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                plt.imshow(img)
                plt.title('this picture has no face in it')
                plt.show()



if __name__=='__main__':
    # 为避免交叉引用，因此仅作为测试的时候引入该模型
    from detector import Model

    # model_path='trained_models_MSELoss_Adam_0.001'
    # parser = argparse.ArgumentParser(description='Detector')
    # parser.add_argument('--save-directory', type=str, default=model_path,
    #                     help='learnt models are saving here')
    # args = parser.parse_args()
    # train_set, test_set = get_train_test_set()
    # valid_loader = torch.utils.data.DataLoader(test_set, batch_size=10)
    # trained_model='detector_epoch_99.pt'
    # model=Model()
    # predict(args, trained_model, model, valid_loader)

    img_path='.\\test_img'
    path='trained_models_SmoothL1Loss_Adam_0.001_2020-03-25-21-26-42'
    file='detector_epoch_99.pt'
    model_path = os.path.join(path,file)
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    my_predict(img_path,model)
