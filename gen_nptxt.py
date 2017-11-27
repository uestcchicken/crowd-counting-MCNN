import numpy as np 
import cv2

IN_PATH = './shanghaitech/part_A_final/train_data/images/'
OUT_PATH = './nptxt/'



for i in range(1, 301):
    img = cv2.imread(IN_PATH + 'IMG_' + str(i) + '.jpg', 0)

    n = np.array(img)
    print('img ', i, ' ', n.shape)
    np.savetxt(OUT_PATH + 'IMG_' + str(i) + '.txt', n)
    
