import scipy.io as sio
import scipy
import scipy.ndimage
import numpy as np
import cv2
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

START_NUM = 197
MAX_NUM = 300
MAT_PATH = './shanghaitech/part_A_final/train_data/ground_truth/'
IMG_PATH = './shanghaitech/part_A_final/train_data/images/'
OUT_PATH = './density/part_A_final/train_data/'

def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    #print('build kdtree...')
    tree = scipy.spatial.KDTree(pts.copy(), leafsize = leafsize)
    #print('query kdtree...')
    distances, locations = tree.query(pts, k = 4)

    print('generating density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype = np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode = 'constant')
    print('done.')
    return density


for img_num in range(START_NUM, MAX_NUM + 1):
    print('image num: ', img_num, ' / ', MAX_NUM)
    matf = MAT_PATH + 'GT_IMG_' + str(img_num) + '.mat'
    data = sio.loadmat(matf)
    pixels = data['image_info'][0][0][0][0][0]

    img = cv2.imread(IMG_PATH + 'IMG_' + str(img_num) + '.jpg', 0)
    shape = img.shape
    print('image shape: ', shape)

    res = np.zeros(shape)
    for p in pixels:
        if int(p[1]) >= shape[0] or int(p[0] >= shape[1]):
            continue
        else:
            res[int(p[1])][int(p[0])] = 1

    den = np.array(gaussian_filter_density(res))

    truth = len(pixels)
    den_sum = np.sum(den)
    error_rate = abs(truth - den_sum) / truth
    print('ground truth: ', truth)
    print('density sum: ', den_sum)
    print('error rate: ', error_rate)
    print('###########################################')
    
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








