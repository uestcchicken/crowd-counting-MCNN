from pyheatmap.heatmap import HeatMap
import numpy as np
import cv2



'''
for img in range(1, 11):
    print(img)
    den = np.loadtxt(open('IMG_' + str(img) + '.csv'), delimiter = ",")
    print(np.max(den))
        
    den = den * 500
    #print(den)
    
    pic = cv2.imread('IMG_' + str(img) + '.jpg', 0)
    w = pic.shape[1]
    h = pic.shape[0]
        
    data = []
    for j in range(len(den)):
        for i in range(len(den[0])):
            for k in range(int(den[j][i])):
                data.append([i + 1, j + 1])
    #print(data)
    hm = HeatMap(data, base = 'IMG_' + str(img) + '.jpg')
    hm.heatmap(save_as = 'heat_' + str(img) + '_with_base.png')
    
    
    hm = HeatMap(data, width = w, height = h)
    hm.heatmap(save_as = 'heat_' + str(img) + '.png')
'''

def heatmap(den, img_num, dataset, info):
    print('generating heat map for img', img_num)
    print('shape:', den.shape)
    
    if info == 'pre':
        den_resized = np.zeros((den.shape[0] * 4, den.shape[1] * 4))
        for i in range(den_resized.shape[0]):
            for j in range(den_resized.shape[1]):
                den_resized[i][j] = den[int(i / 4)][int(j / 4)] / 16
        den = den_resized
    
    
    img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
    count = np.sum(den)
    den = den * 10 / np.max(den)
    img = cv2.imread(img_path + 'IMG_' + str(img_num) + '.jpg', 1)
    
    w = img.shape[1]
    h = img.shape[0]
     
    data = []
    for j in range(len(den)):
        for i in range(len(den[0])):
            for k in range(int(den[j][i])):
                data.append([i + 1, j + 1])
    hm = HeatMap(data, base = img_path + 'IMG_' + str(img_num) + '.jpg')
    hm.heatmap(save_as = 'heat_' + dataset + '_' + str(img_num) + '_' + info + '_' + str(int(count)) + '.png')

    

