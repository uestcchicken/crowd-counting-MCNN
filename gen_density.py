import scipy.io as sio
import scipy
import scipy.ndimage
import numpy as np
import cv2
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

#转换图像范围
START_NUM = 288
MAX_NUM = 300
MAT_PATH = './shanghaitech/part_A_final/train_data/ground_truth/'
IMG_PATH = './shanghaitech/part_A_final/train_data/images/'
OUT_PATH = './density/part_A_final/train_data/'

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    #所有非零点
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    #生成kd树
    leafsize = 1024
    tree = scipy.spatial.KDTree(pts.copy(), leafsize = leafsize)
    #寻找k个最近邻
    distances, locations = tree.query(pts, k = 4)

    print('generating density...')
    #对每个点作高斯核变换
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(gt.shape, dtype = np.float32)
        pt2d[pt[1], pt[0]] = 1.
        #sigma取平均距离的0.3倍
        sigma = (distances[i][0] + distances[i][1] + distances[i][2] + distances[i][3]) * 0.075
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode = 'constant')
    return density


for img_num in range(START_NUM, MAX_NUM + 1):
    print('image num: ', img_num, ' / ', MAX_NUM)
    matf = MAT_PATH + 'GT_IMG_' + str(img_num) + '.mat'
    #读取matlab格式文件
    data = sio.loadmat(matf)
    pixels = data['image_info'][0][0][0][0][0]

    img = cv2.imread(IMG_PATH + 'IMG_' + str(img_num) + '.jpg', 0)
    shape = img.shape
    print('image shape: ', shape)
    #生成标记点阵图
    res = np.zeros(shape)
    for p in pixels:
        if int(p[1]) >= shape[0] or int(p[0] >= shape[1]):
            continue
        else:
            res[int(p[1])][int(p[0])] = 1
    #生成密度图
    den = np.array(gaussian_filter_density(res))
    #检验精确度
    truth = len(pixels)
    den_sum = np.sum(den)
    error_rate = abs(truth - den_sum) / truth
    print('ground truth: ', truth)
    print('density sum: ', den_sum)
    print('error rate: ', error_rate)
    print('###########################################')
    #保存为numpy
    np.savetxt(OUT_PATH + 'DEN_' + str(img_num) + '.txt', den)

    ##绘图部分
    '''
    max = float(np.max(den))
    den = den * 255 / max

    x = []
    y = []
    for i in range(len(den)):
        for j in range(len(den[i])):
            for k in range(int(den[i][j])):
                x.append(j)
                y.append(len(den) - i)
    for i in range(len(den)):
        for j in range(len(den[i])):
            x.append(j)
            y.append(len(den) - i)

    counts, xbins, ybins, image = plt.hist2d(x, y, bins = 100, norm = LogNorm(), cmap = plt.cm.rainbow)
    plt.savefig('density_' + str(img_num) + '.png')
    #plt.show()
    '''
