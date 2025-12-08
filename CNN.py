#import 
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset,random_split
import matplotlib.pyplot as plt
from math import floor
from tqdm.auto import tqdm
from torchvision import datasets, transforms 


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

def calc_output_dim(input_dim , k_size , stride , padding):
        return floor(((input_dim + 2 * padding - k_size) / stride)+ 1)
    
def forward(self, x):
    """Defines how data flows through the network."""
    # Pass through convolutional layers
    x = self.conv_layers(x)      
    # Flatten the output for the dense layers
    x = x.view(x.size(0), -1) # x.size(0) is the batch size
    # Pass through dense layers
    x = self.dense_layers(x)
    return x

def train_model(model, train_loader , optimizer , criterion , num_epochs  , model_name):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    model = model.to(device)
    print(f"Training{model_name} architecture:")
    print(model)
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    
    for epoch in tqdm(range(num_epochs)):

        for _,(images ,labels) in enumerate(train_loader):
            loss = train_batch(images , labels , model ,optimizer , criterion)
            examples_ct += len(images)
            batch_ct += 1
            
def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss
    
def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


def test(model , test_loader):
    model.eval()

    with torch.no_grad():
        correct , total = 0 , 0
        for images, labels in test_loader:
            images,labels = images.to(device) , labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data , 1)
            tottal += label.size(0)
            correct += (predicted == labels).cum().item()
        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})
    #Save the model in exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")

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
    transforms.Resize(16),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                        std =[0.5,0.5,0.5])
])


class ImageDataset(torch.utils.data.Datasets):
    def __init__(self,data_dir,transform=data_transform):
        self.data_dir = data_dir
        self.transform = data_transform
        self.imagepaths = []
        self.labels = []
        self.class_to_idx = {}

        class_names = sorted(os.listdir(data_dir))
        for i , class_name in enumerate(os.listdir(data_dir)):
            class_path = os.path.join(data_dir , class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = i
                #collect all the file paths and corresponding labels
                for file_name in os.listdir(class_path):
                    if file_name.endswith(('.jpg')):
                        self.image_paths.append(os.path.join(class_name , file_name))
                        self.labels.append(i)

    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)
        
    def train_val_data(dataset , val_split_ratio=0.20 ,batch_size):
        #here i will split 20% of train data for validation and hyper parameter fine tuning
        total_size = len(dataset)
        val_size = int(val_split_ratio * total_size)
        train_size = int((1-val_split_ratio) *total_size)
        split_size = [train_size , val_size]
        train_dataset , val_dataset = random_split(
            dataset ,split_size , 
            generator = torch.Generator().manual_seed(42)
        )
        return train_dataset , val_dataset

    def get_loaders(train_dataset, val_dataset, batch_size, num_workers=4):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True, # Always shuffle training data
            num_workers=num_workers 
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False, # Do not shuffle validation data
            num_workers=num_workers
        )
    
        return train_loader, val_loader

    def test_data(dataset,batch_sixe = 64):
        test_loader = DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = False
        )
        return test_dataset


