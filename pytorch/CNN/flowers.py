# author : 'wangzhong';
# date: 07/01/2021 17:16

"""
2.用pytorch进行迁移学习
图像100分类实战
流程：
1. data argumentation
2. dataloader，做批量处理
3. load model
4. 改最后一个fc
5. 训练
6. 保存模型
"""
import copy
import json
from pyexpat import model
import time

import torch
from torch import nn
import torch.optim as optim
import torchvision
import os
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt
import ssl


# 是否继续训练模型的参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # 全部冻结
        for param in model.parameters():
            param.requires_grad = False


def get_device() -> torch.device:
    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


# 选择迁移哪个模型
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        # pretrained 表示是否需要下载
        # model_ft 中包含了该网络的所有特征提取层和全连接层；
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # fc 代表全连接层，取出其输入。全连接层可能是 [2048,1000] 完成 1000 分类，
        # 我们的任务可能是 100 分类，所以需要更改全连接层的维度
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                   nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def flower_start():
    model_ft, input_size = initialize_model("resnet", 102, feature_extract=True, use_pretrained=True)
    device = get_device()
    model_ft = model_ft.to(device)
    print("resnet152 detail: ", model_ft)
    # model.parameters()只返回参数的迭代器，named_parameters()还有网络层的名字
    params = model_ft.named_parameters()
    print("Params need to learn:")
    params_need_update = []
    for param_name, param in params:
        # 之前已经把除了全连接层外的参数的grad设为False了，所以现在只剩下全连接层了
        if param.requires_grad:
            params_need_update.append(param)
            print(param_name)
    data_dir = './flower_data/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    # 读取对应的花名
    with open("cat_to_name.json") as f:
        cat_to_name = json.load(f)

    data_transforms = {
        'train': transforms.Compose(
          [
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
            transforms.CenterCrop(224),  # 从中心开始裁剪，只得到一张图片
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 概率为0.5
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),
            # 迁移学习，用别人的均值和标准差
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
          ]),
        'valid': transforms.Compose(
          [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # 预处理必须和训练集一致
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ]),
    }

    batch_size = 16

    # train和valid的图片，做transform之后用字典保存
    image_datasets = {x: datasets.ImageFolder(
      os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

    # 批量处理，这里都是tensor格式（上面compose）
    dataloaders = {x: torch.utils.data.DataLoader(
      image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"dataset_sizes: {dataset_sizes}")
    
    # 样本数据的标签："1", "2" 数字，json文件中的key值
    class_names = image_datasets['train'].classes
    print(f"class_names: {class_names};")

    # 优化器设置
    optimizer_ft = optim.Adam(params_need_update, lr=1e-2)
    # 学习率衰减：每7个epoch衰减成原来的1/10
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # FC层已经 LogSoftmax() 了，所以不能 nn.CrossEntropyLoss() 来计算了；
    # 因为 nn.CrossEntropyLoss() 相当于 logSoftmax() 和 nn.NLLLoss() 整合
    criterion = nn.NLLLoss()
    # 保存输出的模型
    filename = "wz.pth"

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = wz_model_train(
      model_ft,
      dataloaders,
      criterion,
      optimizer_ft,
      scheduler,
      filename,
      device)

    # 上面只训练了全连接层，我们可以再继续训练所有的参数，进行微调。
    for param in model_ft.parameters():
        param.requires_grad = True

    # 学习率调小一点：我们是在原来基础上微调，学习率大了可能会破坏原来的参数。
    optimizer = optim.Adam(params_need_update, lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 损失函数
    criterion = nn.NLLLoss()
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model_ft.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = wz_model_train(
      model_ft,
      dataloaders,
      criterion,
      optimizer,
      scheduler, filename,
      device,
      num_epochs=5)
    
    # 效果测试
    print("\n======效果测试======\n")

    fig = plt.figure(figsize=(20, 12))
    # batch_size=16，所以行列之积要和 batch 保持一致
    columns = 4
    rows = 4
    dataiter = iter(dataloaders['valid'])
    inputs, classes = next(dataiter)
    output = model_ft(inputs.cuda())
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    print("preds: ", preds)

    for idx in range(columns * rows):
      ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
      # preds[idx] 是标签数字，classes[idx]是正确答案的数字
      ax.set_title(f"{cat_to_name[str(preds[idx])]}, ({cat_to_name[str[classes[idx].item()]]})", 
        color = ("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(classes[idx].item())] else "red"))
      img = transforms.ToPILImage()(inputs[idx])
      plt.imshow(img)
    plt.savefig("valid_data.jpg")


def wz_model_train(model, dataloaders, criterion, optimizer, scheduler, filename: str, device: torch.device, num_epochs=10, is_inception=False):
    start_time = time.time()
    best_acc = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    model.to(device)
    # 保存损失和准确率数据
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 记录每个epoch的learningRate
    LRs = [optimizer.param_groups[0]['lr']]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 清零
                optimizer.zero_grad()
                # 只在训练时计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    # inception会有辅助输出，损失函数为一个线性模型
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # 概率最大的为预测结果
                    _, preds = torch.max(outputs, 1)
                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 计算损失
                # loss计算默认都是取mean，计算批量的loss时，要乘以loss的数量
                # 所以这里计算的是一个epoch里所有样本的loss和正确数量
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # 完整一次的loss均值和准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # 一个epoch里train和valid分别花的时间和loss和准确度
            time_elapsed = time.time() - start_time
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step()
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])

    time_elapsed = time.time() - start_time
    print("\n========")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("========\n")
    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_weights)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    flower_start()