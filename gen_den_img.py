import numpy as np
import cv2
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

for m in range(1,2):
    for l in range(1,2):
        den = np.load('a' + str(m) + str(l) + '.npy')
        den = den.reshape((den.shape[1], den.shape[2]))
        print(den.shape)

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
        plt.savefig('a' + str(m) + str(l) + '.png')
        #plt.show()

        den = np.load('p' + str(m) + str(l) + '.npy')
        den = den.reshape((den.shape[1], den.shape[2]))
        print(den.shape)

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
        plt.savefig('p' + str(m) + str(l) + '.png')
        #plt.show()
