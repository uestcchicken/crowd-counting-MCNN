# Single Image Crowd Counting via Multi Column Convolutional Neural Network

**unfinished** **unfinished** **unfinished**

very ugly unofficial implementation of CVPR2016 paper  
*Single Image Crowd Counting via Multi Column Convolutional Neural Network*  
using **tensorflow**

### installation

1. install tensorflow
2. ```git clone https://github.com/uestcchicken/MCNN.git```

### data setup

All the data setup process follows the pytorch version implementation:   
[svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn#data-setup)

### train 

Set the variable dataset in train.py to 'A' or 'B'  
run ```python3 train.py```  
model is saved to modelA/ or modelB/

### test 

set the variable dataset in test.py to 'A' or 'B'  
run ```python3 test.py```

### result

I've only trained the model for about 2 hours so far on DELL laptop, GTX960m, tensorflow-gpu 1.4.1.  
Maybe the result will be better with more trainging hours. I hope.

A_mae: 135 A_mse: 220  
B_mae: 41  B_mse: 64

In the paper it's:  

A_mae: 110 A_mse: 173  
B_mae: 26  B_mse: 41

Waiting to train on GTX1080.