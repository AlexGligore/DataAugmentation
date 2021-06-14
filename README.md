# Data Augmentation

Small data augmentation module. Test code can be found in data_augmentation.py and in main you can find a small usage example.
Examples:  
### Original:  
![Original](./doc/original.jpg)  
### Flipped:  
![Flipped](./doc/flipped.jpg)  
### Grayscale:  
![Grayscale](./doc/grayscale.jpg)  
### Shifted:  
![Shifted](./doc/shifted.jpg)  

## Operations

Currently the module supports the following opperations:

* Image grayscaling
* Image flipping
* Image rotation
* Image shifting
* Adding image noise (Gaussian)
* Image blurring

## Requirements
* numpy
* scipy
* opencv-python

