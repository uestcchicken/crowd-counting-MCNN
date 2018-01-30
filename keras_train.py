from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import sys
import os 
import cv2
import keras.backend as K
import math

if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 test.py A(or B)')
    exit()
print('dataset:', dataset)

train_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train/'
train_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/train_den/'
val_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val/'
val_den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/val_den/'
img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
den_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'

def data_pre_train():
    print('loading data from dataset ', dataset, '...')
    train_img_names = os.listdir(train_path)
    img_num = len(train_img_names)

    train_data = []
    for i in range(img_num):
        if i % 100 == 0:
            print(i, '/', img_num)
        name = train_img_names[i]
        #print(name + '****************************')
        img = cv2.imread(train_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
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

    print('load data finished.')
    return train_data
    
def data_pre_test():
    print('loading test data from dataset', dataset, '...')
    img_names = os.listdir(img_path)
    img_num = len(img_names)

    data = []
    for i in range(img_num):
        if i % 50 == 0:
            print(i, '/', img_num)
        name = 'IMG_' + str(i + 1) + '.jpg'
        #print(name + '****************************')
        img = cv2.imread(img_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        #print(img.shape)
        den = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))
        #print(den_quarter.shape)
        for i in range(len(den_quarter)):
            for j in range(len(den_quarter[0])):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        #print(den.shape)
        data.append([img, den_quarter])
            
    print('load data finished.')
    return data
    
data = data_pre_train()
data_test = data_pre_test()
np.random.shuffle(data)

x_train = []
y_train = []
for d in data:
    x_train.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_train.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for d in data_test:
    x_test.append(np.reshape(d[0], (d[0].shape[0], d[0].shape[1], 1)))
    y_test.append(np.reshape(d[1], (d[1].shape[0], d[1].shape[1], 1)))
x_test = np.array(x_test)
y_test = np.array(y_test)


def maaae(y_true, y_pred):
    return abs(K.sum(y_true) - K.sum(y_pred))
def mssse(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))


inputs = Input(shape = (None, None, 1))
conv_m = Conv2D(20, (7, 7), padding = 'same', activation = 'relu')(inputs)
conv_m = MaxPooling2D(pool_size = (2, 2))(conv_m)
conv_m = (conv_m)
conv_m = Conv2D(40, (5, 5), padding = 'same', activation = 'relu')(conv_m)
conv_m = MaxPooling2D(pool_size = (2, 2))(conv_m)
conv_m = Conv2D(20, (5, 5), padding = 'same', activation = 'relu')(conv_m)
conv_m = Conv2D(10, (5, 5), padding = 'same', activation = 'relu')(conv_m)
#conv_m = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_m)

conv_s = Conv2D(24, (5, 5), padding = 'same', activation = 'relu')(inputs)
conv_s = MaxPooling2D(pool_size = (2, 2))(conv_s)
conv_s = (conv_s)
conv_s = Conv2D(48, (3, 3), padding = 'same', activation = 'relu')(conv_s)
conv_s = MaxPooling2D(pool_size = (2, 2))(conv_s)
conv_s = Conv2D(24, (3, 3), padding = 'same', activation = 'relu')(conv_s)
conv_s = Conv2D(12, (3, 3), padding = 'same', activation = 'relu')(conv_s)
#conv_s = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_s)

conv_l = Conv2D(16, (9, 9), padding = 'same', activation = 'relu')(inputs)
conv_l = MaxPooling2D(pool_size = (2, 2))(conv_l)
conv_l = (conv_l)
conv_l = Conv2D(32, (7, 7), padding = 'same', activation = 'relu')(conv_l)
conv_l = MaxPooling2D(pool_size = (2, 2))(conv_l)
conv_l = Conv2D(16, (7, 7), padding = 'same', activation = 'relu')(conv_l)
conv_l = Conv2D(8, (7, 7), padding = 'same', activation = 'relu')(conv_l)
#conv_l = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_l)

conv_merge = Concatenate(axis = 3)([conv_m, conv_s, conv_l])
result = Conv2D(1, (1, 1), padding = 'same')(conv_merge)
'''

inputs = Input(shape = (None, None, 1))
conv_m = Conv2D(20, (7, 7), padding = 'same', activation = 'relu')(inputs)
conv_m = MaxPooling2D(pool_size = (2, 2))(conv_m)
conv_m = (conv_m)
conv_m = Conv2D(40, (5, 5), padding = 'same', activation = 'relu')(conv_m)
conv_m = MaxPooling2D(pool_size = (2, 2))(conv_m)
conv_m = Conv2D(20, (5, 5), padding = 'same', activation = 'relu')(conv_m)
conv_m = Conv2D(10, (5, 5), padding = 'same', activation = 'relu')(conv_m)
#conv_m = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_m)

conv_s = Conv2D(24, (5, 5), padding = 'same', activation = 'relu')(inputs)
conv_s = MaxPooling2D(pool_size = (2, 2))(conv_s)
conv_s = (conv_s)
conv_s = Conv2D(48, (3, 3), padding = 'same', activation = 'relu')(conv_s)
conv_s = MaxPooling2D(pool_size = (2, 2))(conv_s)
conv_s = Conv2D(24, (3, 3), padding = 'same', activation = 'relu')(conv_s)
conv_s = Conv2D(12, (3, 3), padding = 'same', activation = 'relu')(conv_s)
#conv_s = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_s)

conv_l = Conv2D(16, (9, 9), padding = 'same', activation = 'relu')(inputs)
conv_l = MaxPooling2D(pool_size = (2, 2))(conv_l)
conv_l = (conv_l)
conv_l = Conv2D(32, (7, 7), padding = 'same', activation = 'relu')(conv_l)
conv_l = MaxPooling2D(pool_size = (2, 2))(conv_l)
conv_l = Conv2D(16, (7, 7), padding = 'same', activation = 'relu')(conv_l)
conv_l = Conv2D(8, (7, 7), padding = 'same', activation = 'relu')(conv_l)
#conv_l = Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(conv_l)

conv_merge = Concatenate(axis = 3)([conv_m, conv_s, conv_l])
result = Conv2D(1, (1, 1), padding = 'same')(conv_merge)

'''

model = Model(inputs = inputs, outputs = result)

adam = Adam(lr = 1e-4)
model.compile(loss = 'mse', optimizer = adam, metrics = [maaae, mssse])


best_mae = 10000
best_mae_mse = 10000
best_mse = 10000
best_mse_mae = 10000
for i in range(200):
    model.fit(x_train, y_train, epochs = 3, batch_size = 1, validation_split = 0.2)

    score = model.evaluate(x_test, y_test, batch_size = 1)
    score[2] = math.sqrt(score[2])
    print(score)
    if score[1] < best_mae:
        best_mae = score[1]
        best_mae_mse = score[2]
        
        json_string = model.to_json()
        open('model.json', 'w').write(json_string)
        model.save_weights('weights.h5')
    if score[2] < best_mse:
        best_mse = score[2]
        best_mse_mae = score[1]

    print('best mae: ', best_mae, '(', best_mae_mse, ')')
    print('best mse: ', '(', best_mse_mae, ')', best_mse)
    





