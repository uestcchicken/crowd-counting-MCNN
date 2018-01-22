# Single Image Crowd Counting via Multi Column Convolutional Neural Network

very ugly unofficial implementation of CVPR2016 paper  
*Single Image Crowd Counting via Multi Column Convolutional Neural Network*  
using **tensorflow**

### installation

1. install tensorflow
2. ```git clone https://github.com/uestcchicken/MCNN.git```

### data setup

All the data setup process follows the pytorch version implementation:   
[svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn)

### train 

Set the variable dataset in train.py to 'A' or 'B'  
run ```python3 train.py```  
model is saved to modelA/ or modelB/

### test 

set the variable dataset in test.py to 'A' or 'B'  
run ```python3 test.py```

### result

I've only trained for about 1 hour so far on DELL laptop, GTX960m.
Maybe the result will be better with more trainging hours. I hope.

A_mae: 135 A_mse: 220
B_mae: B_mse:

Waiting to train on GTX1080.