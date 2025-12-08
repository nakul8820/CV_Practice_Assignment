#import 
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.Functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import math
import torchvision import datasets, transforms


'''
1 . Build a small CNN model consisting of 555 convolution layers. Each convolution layer would be 
followed by an activation and a max-pooling layer.

2 . After 5 such conv-activation-maxpool blocks, you should have one dense layer followed by the 
output layer containing 10  neurons (1 for each of the 10 classes). 

3 . The input layer should be compatible with the images in the iNaturalist dataset dataset.

4 . The code should be flexible such that the number of filters, size of filters, and activation function 
    of the convolution layers  and dense layers can be changed. 
    You should also be able to change the number of neurons in the dense layer.
'''
class CNNModel(nn.Module):
  def __init__(self,
               input_size ,            #input HXW
               conv_out_channels  ,     #list of int
               kernel_size  ,           #list of int
               dense_layer_out,        #integer
               activation_func  , # expected Passed as nn.Module class (e.g., nn.ReLU , nn.Sigmoid etc)
               dense_layer_func ,
               num_of_class):
    super(CNNModel,self).__init__()
    self.conv_layers = nn.Sequential()
    in_channels = 3 #RGB Assumption

    if not (len(kernel_size) == len(conv_out_channels)):
        raise ValueError("Kernel Size and Conv output channels length should be same")

    def calc_output_dim(input_dim , k_size , stride , padding):
        return floor(((input_dim + 2 * padding - k_size) / stride)+ 1)
    
    current_hw_dim = input_size
    k_size = 0
    pool_k_size = 2
    pool_stride = 2
    pool_padding = 2
    for i ,(k_size , out_channels) in enumerate(zip(kernel_size,conv_out_channels)):
        self.conv_layers.add_module(f'conv{i+1}', nn.Conv2d(in_channels ,
                                                             out_channels ,
                                                             k_size,
                                                            padding = "same"))
        
        self.conv_layers.add_module(f'activation_{i+1}', activation_func())
        self.conv_layers.add_module(f'pooling_{i+1}', nn.MaxPool2d(kernel_size = 2, padding=2))

        current_hw_dim = calc_output_dim(current_hw_dim, pool_k_size, pool_stride, pool_padding)
        in_channels = out_channels  #output channels of previous layer becomes input for next channels
    input_for_dense_layer = current_hw_dim * current_hw_dim * in_channels 
    self.dense_layers = nn.Sequential()

    self.dense_layers.add_module('fc1' , nn.Linear(input_for_dense_layer , dense_layer_out))
    self.dense_layers.add_module(f'activation_',dense_layer_func())
    #output layer
    self.dense_layers.add_module('output' , nn.Linear(dense_layer_out , num_of_class))

def forward(self, x):
    """Defines how data flows through the network."""
    # Pass through convolutional layers
    x = self.conv_layers(x)
        
    # Flatten the output for the dense layers
    x = x.view(x.size(0), -1) # x.size(0) is the batch size

    # Pass through dense layers
    x = self.dense_layers(x)

    return x
'''
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

data_transform = transforms.Compose([
    transforms.Resize(256),
    transform.ToTensor(),
    transform.Normalize(mean=[0.5,0.5,0.5],
                        std =[0.5,0.5,0.5]
])


class dataloader(Datasets):
    def __init__(self,data_dir,transform=data_transform):
        self.data_dir = data.dir
        self.transform = data_transform
        self.imagepaths = []
        self.labels = []
        sel.class_to_idx = {}

        

