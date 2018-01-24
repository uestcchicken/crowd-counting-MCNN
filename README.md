# Single Image Crowd Counting via Multi Column Convolutional Neural Network

**unfinished**  
**unfinished**  
**unfinished**

very ugly unofficial implementation of CVPR2016 paper  
[Single Image Crowd Counting via Multi Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)  
using **tensorflow**

### installation

1. install tensorflow
2. ```git clone https://github.com/uestcchicken/MCNN.git```

### data setup

All the data setup process follows the pytorch version implementation:   
[svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn#data-setup)

### train 

run ```python3 train.py A(or B)```  
model is saved to modelA/ or modelB/

### test 

run ```python3 test.py A(or B)```

### result

The model here is trained on DELL laptop, GTX960m, tensorflow-gpu 1.4.1.  

A_mae: 119 A_mse: 188  
B_mae: 32  B_mse: 55

In the paper it's:  

A_mae: 110 A_mse: 173  
B_mae: 26  B_mse: 41
