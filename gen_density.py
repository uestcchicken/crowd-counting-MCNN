import scipy.io as sio
import scipy
import scipy.ndimage
import numpy as np
import cv2
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

PICTURE_NUM = 4

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    print(gt_count)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    print('build kdtree...')
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    print('query kdtree...')
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            print(sigma)
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

matf = 'GT_IMG_' + str(PICTURE_NUM) + '.mat'
data = sio.loadmat(matf)
pixels = data['image_info'][0][0][0][0][0]
print(pixels)

img = cv2.imread('IMG_' + str(PICTURE_NUM) + '.jpg', 0)
print(img.shape)

res = np.zeros(img.shape)
print(res.shape)

for p in pixels:
    res[int(p[1])][int(p[0])] = 1
    
counter = 0
for i in res:
    for j in i:
        if j == 1:
            counter += 1
            
print(counter)
print(len(pixels))

den = np.array(gaussian_filter_density(res))
print(den)
print(den.shape)

print("truth: ", len(pixels))
print("sum: ", np.sum(den))

max = np.max(den)
for i in range(len(den)):
    for j in range(len(den[i])):
        den[i][j] = int(den[i][j] * 255 / max)

print(den)
print(np.max(den))

x = []
y = []

for i in range(len(den)):
    for j in range(len(den[i])):
        for k in range(int(den[i][j])):
            x.append(j)
            y.append(len(den) - i)
            
print(len(x))
print(len(y))

for i in range(len(den)):
    for j in range(len(den[i])):
        x.append(j)
        y.append(len(den) - i)

counts,xbins,ybins,image = plt.hist2d(x,y,bins=100,norm=LogNorm(), cmap = plt.cm.rainbow)
plt.savefig('density_' + str(PICTURE_NUM) + '.png')
plt.show()









