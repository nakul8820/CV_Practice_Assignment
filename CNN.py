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
  def __init__(self,
               input_size ,            #input HXW
               conv_out_channels ,     #list of int
               kernel_size ,           #list of int
               dense_layer_out,        #integer
               activation_func= nn.ReLU , # Passed as nn.Module class (e.g., nn.ReLU)
               dense_layer_func= nn.ReLU,
               num_of_class):
    super(CNNModel,self).__init__()
    conv_layers = []
    channels = 3 #RGB 
                   
    for (conv_out , k_size) in zip(conv_out_channels , kernel_size):
        conv_layers.append(
            nn.Conv2d(in_channels=channels,
                      out_channels = conv_out,
                      kernel_size = k_size,
                      padding="same"
                     )
        )
        #Activation Function
        conv_layers.append(activation_func())

        #max pooling(2X2 downsampling) 
        conv_layers.append(nn.MaxPoo2d(kernel_size=2 , stride=2)

        #update for chanel count for next layer
        channels = conv_out
    self.conv_sequence = nn.Sequential(*conv_layers)
    '''
    def forward(self,x):
        if self.activation_func == "relu":
            for i in range(1,len(self.conv_out)+1):
                x = 
    self.conv1 = nn.Conv2d(3 ,32 , kernel_size)    #parameters
    self.conv2 = nn.Conv2d(32 ,64 , kernel_size)    #1.in_channels 
    self.conv3 = nn.Conv2d(64 ,128 , kernel_size)    #2.out_channels
    self.conv4 = nn.Conv2d(128 ,256 , kernel_size)    #3.Kernels
    self.conv5 = nn.Conv2d(256,512 , kernel_size)
    self.fc1 = nn.Linear(in_features=512 , out_features=dense_layer_out)
    self.fc2 = nn.Linear(in_fearures=

def forward(self , x , conv_activation_func , dense_activation_func):
    x = F.conv_activation_func(self.conv1(x))
    x = F.max_pool2d(x , 2)
    x = F.conv_activation_func(self.conv2(x))
    x = F.max_pool2d(x , 2)
    x = F.conv_activation_func(self.conv3(x))
    x = F.max_pool2d(x , 2)
    x = F.conv_activation_func(self.conv4(x))
    x = F.max_pool2d(x , 2)
    x = F.conv_activation_func(self.conv5(x))
    x = F.max_pool2d(x , 2)
    '''
