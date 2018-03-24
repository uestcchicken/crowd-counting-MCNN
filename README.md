# Single Image Crowd Counting via Multi Column Convolutional Neural Network

**unfinished**  
**unfinished**  
**unfinished**

A very ugly unofficial implementation of CVPR2016 paper  
[Single Image Crowd Counting via Multi Column Convolutional Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)  
using **tensorflow** and **keras**(only for B)

### installation

1. Install tensorflow (and keras)
2. ```git clone https://github.com/uestcchicken/MCNN.git```

### data setup

All the data setup process follows the pytorch version implementation:   
[svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn#data-setup)

### train
For tensorflow:  
run ```python3 train.py A(or B)```  
model is saved to modelA/ or modelB/

For keras:  
run ```python3 keras_train.py B```  
model is saved to keras_modelB/

### test 

For tensorflow:  
run ```python3 test.py A(or B)```  

For keras:  
run ```python3 keras_test.py B```

(uncomment code containing heatmap in network.py to generate heatmap)

### result

The model here is trained on DELL laptop, GTX960m, tensorflow-gpu 1.4.1.  

A_mae: 119 A_mse: 188  
B_mae: 32  B_mse: 55

Using keras:

B_mae: 29  B_mse: 47

In the paper it's:  

A_mae: 110 A_mse: 173  
B_mae: 26  B_mse: 41

### heatmap

actual: 1110  
![](samples/heat_A_2_act_1110.png)  
predicted: 1246  
![](samples/heat_A_2_pre_1246.png)


