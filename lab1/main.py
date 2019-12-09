# -*- coding:utf-8 -*-
# @time : 2019.12.02
# @IDE : pycharm
# @author : wangzhebufangqi
# @github : https://github.com/wangzhebufangqi

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import models
import config

train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),#随机裁剪到256*256
        transforms.RandomRotation(degrees=15),#随机旋转
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.CenterCrop(size=224),#中心裁剪到224*224
        transforms.ToTensor(),#转化成张量
        transforms.Normalize([0.485, 0.456, 0.406],#归一化
                             [0.229, 0.224, 0.225])
])

test_valid_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])


train_directory = config.TRAIN_DATASET_DIR
valid_directory = config.VALID_DATASET_DIR

batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES


train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
train_data_size = len(train_datasets)
train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

valid_datasets = datasets.ImageFolder(valid_directory,transform=test_valid_transforms)
valid_data_size = len(valid_datasets)
valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size)

resnet50 = models.resnet50(pretrained=True)
print('before:{%s}\n'%resnet50)
for param in resnet50.parameters():
    param.requires_grad = False
fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1)
)
print('after:{%s}\n'%resnet50)

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())


def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#若有gpu可用则用gpu
    record = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):#训练epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()#训练

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)
            # 记得清零
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()#验证

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  :#记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        #torch.save(model, 'trained_models/vehicle-10_model_' + str(epoch + 1) + '.pth')
    return model, record


if __name__=='__main__':
    num_epochs = config.NUM_EPOCHS
    trained_model, record = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
    torch.save(record, config.TRAINED_MODEL)

    record = np.array(record)
    plt.plot(record[:, 0:2])
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('loss.png')
    plt.show()

    plt.plot(record[:, 2:4])
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')
    plt.show()
