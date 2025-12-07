#import 
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.Functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

'''
1 . Build a small CNN model consisting of 555 convolution layers. Each convolution layer would be 
followed by an activation and a max-pooling layer.

2 . After 555 such conv-activation-maxpool blocks, you should have one dense layer followed by the 
output layer containing 10 10 10 neurons (111 for each of the 101010 classes). 

3 . The input layer should be compatible with the images in the iNaturalist dataset dataset.

4 . The code should be flexible such that the number of filters, size of filters, and activation function 
    of the convolution layers  and dense layers can be changed. 
    You should also be able to change the number of neurons in the dense layer.
'''
class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel,slef).__init__()
    self.conv1 = nn.Conv2d()
