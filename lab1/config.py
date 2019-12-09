# -*- coding:utf-8 -*-
# @time : 2019.12.02
# @IDE : pycharm
# @author : wangzhebufangqi
# @github : https://github.com/wangzhebufangqi

#数据集的类别
NUM_CLASSES = 10

#训练时batch的大小
BATCH_SIZE = 32

#训练轮数
NUM_EPOCHS= 25

##预训练模型的存放位置
#下载地址：https://download.pytorch.org/models/resnet50-19c8e357.pth
PRETRAINED_MODEL = './resnet50-19c8e357.pth'

##训练完成，权重文件的保存路径,默认保存在trained_models下
TRAINED_MODEL = 'trained_models/vehicle-10_record.pth'

#数据集的存放位置
TRAIN_DATASET_DIR = './vehicle-10/train'
VALID_DATASET_DIR = './vehicle-10/val'
