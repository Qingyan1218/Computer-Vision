from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import os
import time
from data import get_train_test_set,load_data
from predict import predict, my_predict
import torchvision
torch.set_default_tensor_type(torch.FloatTensor)

# # 网络搭建
# # 经过试验，采用batchnorm，不采用dropout效果最好
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # # 搭建网络，卷册层和全连接层分开表示，层内不同的模块分开表示
        self.Conv=torch.nn.Sequential(
            torch.nn.Conv2d(1,8,kernel_size=5,stride=2,padding=0),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.AvgPool2d(kernel_size=2,stride=2,ceil_mode=True),

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

        self.FullConnection=torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 80, 128),
            torch.nn.PReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 128),
            torch.nn.PReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 42)
        )

        self.JudgeFaceConv=torch.nn.Sequential(
            torch.nn.Conv2d(80, 40, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(40),
            # torch.nn.Dropout(p=0.5),
        )

        self.JudgeFaceFull=torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 40,128),
            torch.nn.PReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 128),
            torch.nn.PReLU(),
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        x=self.Conv(x)
        # 分支1,label
        x1=self.JudgeFaceConv(x)
        # x1.shape的shape为([64, 40, 4, 4]),所以view的时候是4x4x40
        x1=x1.view(-1, 4 * 4 * 40)
        x1=self.JudgeFaceFull(x1)


        # 与全连接层连接前需要将数据flatten
        # 分支2,keypoints
        x2=x.view(-1, 4 * 4 * 80)
        x2=self.FullConnection(x2)
        return x1,x2


def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # 保存模型
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    # label和pts的loss函数
    label_critreion=criterion[0]
    pts_criterion = criterion[1]

    Loss_list = {'train': [], 'val': []}
    Accuracy_list_label = {'train': [], 'val': []}

    train_quantity=len(train_loader.dataset)
    valid_quantity=len(valid_loader.dataset)
    print(f'train_quantity: {train_quantity},valid_quantity: {valid_quantity}')

    for epoch_id in range(epoch):
        # 模型训练
        model.train()
        correct_train_labels = 0
        correct_valid_labels=0

        train_TP = 0
        train_FP = 0
        train_FN = 0
        valid_TP = 0
        valid_FP = 0
        valid_FN = 0

        running_loss=0
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']
            label=batch['label']

            # ground truth，真实数据
            input_img = img.to(device)
            target_pts = landmark.to(device)
            target_label = label.to(device)

            # clear the gradients of all optimized variables，梯度清零
            optimizer.zero_grad()
			
            # get output，获取输出
            output_pts= model(input_img)[1]
            output_label=model(input_img)[0]

            # get loss，获得损失
            # 对于crossentropyloss，他会自己进行onehot，另外要将形状变成(batchsize,)
            loss0=label_critreion(output_label,torch.squeeze(target_label))
            _,preds=torch.max(output_label,1)

            # 累计每次预测正确的标签数量,preds的shape是一行，target_label的shape是batch_size行
            target_label=target_label.view(-1, )

            # # label为0的地方无需计算loss，对于label为1的地方计算loss
            mask= target_label==1
            loss1=pts_criterion(output_pts[mask],target_pts[mask])
            # loss1 = pts_criterion(output_pts, target_pts)
            loss=loss0+loss1

            # 反传，梯度更新
            loss.backward()
            optimizer.step()

            correct_train_labels += torch.sum(preds == target_label)
            # 将预测正确的标签取出来
            TP_add_TN = preds[preds==target_label]
            # 其中为1的是TP，剩下的是TN
            train_TP += len(TP_add_TN[TP_add_TN==1])
            # 将预测不正确的标签取出来
            FP_add_FN = preds[preds != target_label]
            # 预测为正，实际为负的样本是FP
            train_FP += len(FP_add_FN[FP_add_FN==0])
            # 预测为负，实际为正的样本是FN
            train_FN += len(FP_add_FN[FP_add_FN==1])
            running_loss += loss.item() # * img.size(0)
            # 打印信息
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}'.format(
				epoch_id, batch_idx * len(img), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

        # 下面的式子计算一些模型的评价指标
        epoch_loss = running_loss / train_quantity
        Loss_list['train'].append(epoch_loss)
        epoch_acc_label = correct_train_labels.double() *100/train_quantity
        train_precision=train_TP*100/(train_TP+train_FP+1e-8)
        train_recall=train_TP*100/(train_TP+train_FN+1e-8)
        print(f'train_TP:{train_TP},train_FP:{train_FP}, train_FN:{train_FN}')
        print(f'train_accuracy:{epoch_acc_label:0.4f}%,train_precision:{train_precision:0.4f}%,train_recall:{train_recall:0.4f}%')
        Accuracy_list_label['train'].append(100 * epoch_acc_label)

        # 模型验证
        valid_mean_pts_loss = 0.0
        model.eval()

        with torch.no_grad():
            # 用于记录测试批次个数
            valid_batch_cnt = 0

            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                valid_landmark = batch['landmarks']
                valid_label = batch['label']

                input_img = valid_img.to(device)
                target_pts = valid_landmark.to(device)
                target_label = valid_label.to(device)

                output_pts = model(input_img)[1]
                output_label = model(input_img)[0]
                _, preds = torch.max(output_label, 1)
                # 每一批的损失
                valid_loss0 = label_critreion(output_label,torch.squeeze(target_label))
                target_label = target_label.view(-1, )
                correct_valid_labels += torch.sum(preds == target_label)


                mask = target_label == 1
                valid_loss1 = pts_criterion(output_pts[mask], target_pts[mask])
                # valid_loss1 = pts_criterion(output_pts, target_pts)

                # 所有批次的总损失
                valid_loss = valid_loss0 + valid_loss1
                valid_mean_pts_loss += valid_loss.item()

                TP_add_TN = preds[preds == target_label]
                valid_TP += len(TP_add_TN[TP_add_TN == 1])
                # 将预测不正确的标签取出来
                FP_add_FN = preds[preds != target_label]
                # 预测为正，实际为负的样本是FP
                valid_FP += len(FP_add_FN[FP_add_FN == 0])
                # 预测为负，实际为正的样本是FN
                valid_FN += len(FP_add_FN[FP_add_FN == 1])

            # 所有批次的平均损失，下面的式子计算一些模型的评价指标
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            epoch_loss = running_loss / valid_quantity
            Loss_list['val'].append(epoch_loss)
            epoch_acc_label = correct_valid_labels.double()*100/ valid_quantity
            valid_precision = valid_TP *100/ (valid_TP + valid_FP+1e-8)
            valid_recall = valid_TP *100/ (valid_TP + valid_FN+1e-8)
            print(f'valid_TP:{valid_TP},valid_FP:{valid_FP}, valid_FN:{valid_FN}')
            print(f'valid_accuracy:{epoch_acc_label:0.4f}%,valid_precision:{valid_precision:0.4f}%,valid_recall:{valid_recall:0.4f}%')
            Accuracy_list_label['val'].append(100 * epoch_acc_label)
            print('Valid: loss: {:.6f}'.format(valid_mean_pts_loss))
        print('====================================================')

        # 保存模型
        if args.save_model:
            if (epoch_id+1)%10 == 0:
                saved_model_name = os.path.join(args.save_directory,
                                            'detector_epoch' + '_' + str(epoch_id) + '.pt')

                torch.save(model.state_dict(), saved_model_name)
        if args.save_model:
            # 保存整个模型
            save_model_name = os.path.join(args.save_directory,args.save_directory + '.pth')
            torch.save(model,save_model_name)
    return loss,valid_mean_pts_loss

def main_test(phase):
    parser = argparse.ArgumentParser(description='Detector')
    # 当参数名称中间带'-'时，提取变量要变成’_‘
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',				
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',		
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.3, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',   # Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args()
    # # 以下定义以下通用的参数
    torch.manual_seed(args.seed)
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    # cuda:0
    # For multi GPUs, nothing need to change here
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
	
    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

    args.phase=phase

    # # 训练
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Building Model')
        model = Model().to(device)

        # # 均方误差
        # criterion_pts = torch.nn.MSELoss()
        criterion_pts = torch.nn.SmoothL1Loss()
        # # 二分类最后可以只输出1维，然后使用BCELoss()或BCEWithLogitsLoss()
        criterion_label = torch.nn.CrossEntropyLoss()
        criterion=[criterion_label,criterion_pts]


        # # SGD优化，总是出现loss为NAN，因为计算值与真实值差距太大了，约为100倍，
        # # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # # 采用Adam时可以不用缩小真实值，能够快速收敛
        optimizer = optim.Adam(model.parameters(),lr=0.001)
        # # RMSprop和SGD一样，不缩放容易爆表
        # # optimizer = optim.RMSprop(model.parameters())

        # # 新建一个文件夹，命名方式为损失函数+优化器+学习率+当前日期
        args.save_directory = 'trained_models_' + criterion_pts.__class__.__name__ + '_' \
                              + optimizer.__class__.__name__ + '_' + str(args.lr)+'_'\
                              +time.strftime('%Y-%m-%d-%H-%M-%S')

        print('===> Start Training')
        train_losses, valid_losses = \
			train(args, train_loader, valid_loader, model, criterion, optimizer, device)
        print(train_losses,valid_losses)
        print('====================================================')

    # # 测试
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Building Model')
        model_path='trained_models_SmoothL1Loss_Adam_0.001_2020-03-28-15-11-28'
        args.save_directory=model_path

        # # 使用模型
        trained_model='detector_epoch_99.pt'
        # # model.eval()在predict函数中进行
        model=Model()

        print('===> Test')
        predict(args, trained_model, model, valid_loader)

    # # 微调
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Building Model')
        # # 首先加载需要微调的模型
        path = 'trained_models_SmoothL1Loss_Adam_0.001_2020-03-28-15-11-28'
        # # 使用模型
        file = 'detector_epoch_99.pt'
        model_path = os.path.join(path, file)

        model = Model()
        model.load_state_dict(torch.load(model_path))
        # # 首先对冻结之前训练的参数
        for param in model.parameters():
            param.requires_grad = False
        # # 对于需要调整的网络层重新定义，默认参数需要更新
        model.FullConnection = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 80, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 42)
        )
        model.to(device)

        print('===> Finetune')
        criterion_pts = torch.nn.MSELoss()
        criterion_label = torch.nn.CrossEntropyLoss()
        criterion=[criterion_label,criterion_pts]
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # 新建一个文件夹，命名方式为损失函数+优化器+学习率+Finetune
        args.save_directory = 'trained_models_' + criterion_pts.__class__.__name__ + '_' \
                              + optimizer.__class__.__name__ + '_' + str(args.lr)+'_'+'Finetune'

        train_losses, valid_losses = \
            train(args, train_loader, valid_loader, model, criterion, optimizer, device)
        print(train_losses,valid_losses)
        print('====================================================')

     # # 预测
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Building Model')
        img_path = '.\\test_img'
        path = 'trained_models_SmoothL1Loss_Adam_0.001_2020-03-28-15-11-28'
        file = 'detector_epoch_99.pt'
        model_path = os.path.join(path, file)
        model = Model()
        model.load_state_dict(torch.load(model_path))

        # # 如果模型是finetune的，那么相关的层需要改变，或者直接加载整个模型
        # file='trained_models_MSELoss_SGD_0.001_Finetune.pth'
        # model.load(file)
        model.eval()

        print('===> Predict')
        # # my_predict是针对文件夹中的图片直接拿来预测的
        my_predict(img_path, model)

if __name__ == '__main__':
    phase_list=['train','test','finetune','predict']
    main_test(phase_list[0])

