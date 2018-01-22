import numpy as np 
import cv2
import tensorflow as tf
import os
import random

LEARNING_RATE = 1e-4
BATCH_SIZE = 1
EPOCH = 50000

train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train/'
train_den_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den/'
val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val/'
val_den_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den/'


def data_pre():
    print('loading data...')
    train_img_names = os.listdir(train_path)
    img_num = len(train_img_names)

    train_data = []
    for i in range(20):
        if i % 100 == 0:
            print(i, '/', img_num)
        
        name = train_img_names[i]
        #print(name + '****************************')
        img = cv2.imread(train_path + name, 0)
        #print(img.shape)
        
        den = np.loadtxt(open(train_den_path + name[:-4] + '.csv'), delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        #print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        
        train_data.append([img, den_quarter])

    #print(len(train_data))
    #print(train_data[0])
    #print(train_data[0][0])
    print('load data finished.')
    
    s = 0
    for d in train_data:
        print(np.sum(d[1]))
        s += np.sum(d[1])
    print(s / 20)
    
    return train_data
    
data_pre()