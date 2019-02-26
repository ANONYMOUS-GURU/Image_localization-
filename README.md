# Image_localization-
The flipkart image localization challenge.


## Test Results
![](images/Selection_007.png)
![](images/Selection_008.png)
![](images/Selection_009.png)
![](images/Selection_010.png)



## Details

### 1. Preprocessing:

Median filter of all images were taken 3 times with a filter size of 5.
Then Canny filter of all images were taken and were dilated(10 iters) and eroded(15 iters) to make filters from image.This single channel image was then stacked with the 3 channeled input image(RGB) after the median filter.

### 2. Network Architecture:

The network was trained using first a feature extractor(CNN) and then a classifier (fc layers) . Firstly two conv layers were used to compress the image. Then these layers were passed on to the next 3 conv layers.
The output of the last 4 conv layers were stacked and then fed to an average pooling layer. This layer stacked the entire receptive field to 1x1 size.
After stacking these layers all the values were then fed to a fc layer which was further fed to another fc layer.

### 3. Loss function:

The loss fuction is the weighted sum of l2 loss and IoU loss. As IoU loss being non-differentiable we started of with a very large weight of l2 loss and it was decremented with increase in the training batches.


### 4. Training:

The entire model was trained using learning rate starting from 0.05 and was decreased to 6.904e-5 at steps of 1.414  (i.e 2^0.5)  for 19 epochs.No validation set was used during final training.

### 5. Results:

Training IoU -- 0.78
Training l2 --  1456.77

Test IoU -- 0.743

### Tensorboard Visualisation:
IoU score:
![](images/Selection_016.png)

L2 Loss:
![](images/Selection_017.png)


