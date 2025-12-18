pip install wandb 

import wandb
wandb.login(key=wandb_api_key)


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,random_split,Dataset
from torchvision import transforms
from torch.nn import Sequential , LazyLinear
from torchvision.datasets import ImageFolder
import math


# **1 . Build a small CNN model consisting of 555 convolution layers. Each convolution layer would be followed by an activation and a max-pooling layer.2 . After 5 such conv-activation-maxpool blocks, you should have one dense layer followed by the output layer containing 10  neurons (1 for each of the 10 classes). 3 . The input layer should be compatible with the images in the iNaturalist dataset dataset.4 . The code should be flexible such that the number of filters, size of filters, and activation function of the convolution layers  and dense layers can be changed. You should also be able to change the number of neurons in the dense layer.**


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sweep_config = {
    'method': 'bayes' ,
    
    'metric': {
        'name': 'test_accuracy', # Track a specific metric name from your training loop
        'goal': 'maximize'   
    },
    'parameters': {
        # --- Model Architecture: Convolutional Layers ---
        'conv_out_channels': {
            # Defines how 'base_filters' scale (your script must handle this logic)
            'values': [ 'same_64', 'same_128', 'double_from_64','double_from_128']
        },
        'kernel_size': {
            # Allows sweeping a single kernel size used across all layers (e.g. all 3x3 or all 5x5)
            'values': [3, 5] 
        },
        'activation_func': {
            # The activation used in the conv block (passed to your class as an object)
            'values': ["ReLU", "GELU", "SiLU"]
        },
        #'batch_normalization': {
        #    'values': ["yes","no"]
        #},
        'dense_layer_out': {
            'values': [256,512]
        },
        
        'dense_layer_func': {
            # The activation used in the dense block (passed to your class as an object)
            'values': ["Sigmoid"]
        },
        
        # Regularization/Training Hyperparameters
        'drop_out_input': {
            # Dropout applied after the flatten step, before the first dense layer
            'values': [0.1, 0.2,0.0]
        },
        'drop_out_hidden': {
            # Dropout applied between dense layers
            'values': [0.2, 0.3, 0.4]
        },
        'optimizer': {
            'values': ['adam'] #, 'sgd', 'rmsprop']
        },
        'learning_rate': {
            'distribution':  'uniform' , #uniform
            'min': 0.00001,
            'max': 0.0001
        },
        'batch_size': {
            'values': [128 , 256]
        },
        
        'num_of_class': {
            'value': 10 
        },
        'epochs': {
            'value': 10
        }
    }
}


sweep_id = wandb.sweep(sweep_config, project="CV_project")

wandb.agent(sweep_id, train_test, count=10)
test_accuracy = test_model(model , 64)
wandb.log({"test_accuracy":test_accuracy })
print("Test_accuracy",test_accuracy)

model = None

class CNNModel(nn.Module):
    def __init__(self,
                conv_out_channels,
                kernel_size,     
                dense_layer_out,
                activation_func ,
                dense_layer_func ,
                num_of_class,
                drop_out_input,
                drop_out_hidden
                ):
        super(CNNModel,self).__init__()
        
        self.conv_layers = Sequential()
        in_channels = 3 #RGB 
        
        if len(conv_out_channels) != len(kernel_size):
            raise ValueError("Kernels and Channels length should be samex")
        
        pool_kernel,pool_stride, pool_pad = 2 , 2 , 0
        
        drop_out_ratio_input = drop_out_input or 0.1
        self.conv_layers.add_module("dropout_input",nn.Dropout(p=drop_out_ratio_input))
        
        for i , (k_size,out_channels) in enumerate(zip(kernel_size , conv_out_channels)):
            self.conv_layers.add_module(f'conv{i+1}',nn.Conv2d(in_channels,
                                                              out_channels,
                                                              k_size,
                                                              padding='same'))
            self.conv_layers.add_module(f'activ{i+1}',activation_func())
            self.conv_layers.add_module(f'pooling{i+1}',nn.MaxPool2d(pool_kernel,
                                                                    pool_stride))
            self.conv_layers.add_module(f'dropout{i+1}',nn.Dropout2d(p=drop_out_hidden))
            in_channels = out_channels
        
        self.dense_layers = Sequential()
    
        self.dense_layers.add_module("fc1",nn.LazyLinear(out_features=dense_layer_out) )
        self.dense_layers.add_module("activ_dense", dense_layer_func())
        self.dense_layers.add_module('output',nn.Linear(dense_layer_out ,num_of_class))

    def forward(self,x):
        x = self.conv_layers(x)
        x = torch.flatten(x,1)
        x = self.dense_layers(x)
        return x

def get_activation_function(name):
    if name == "ReLU":
        return nn.ReLU
    elif name == "GELU":
        return nn.GELU
    elif name == "Sigmoid":
        return nn.Sigmoid
    elif name == "SiLU":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_conv_out_channels(value:str):
    if value == 'same_32':
        return [32,32,32,32,32]
    elif value == 'same_64':
        return [64,64,64,64,64]
    elif value == 'double_from_16':
        return [16,32,64,128,256]
    elif value == 'double_from_32':
        return [32,64,128,256,512]
    else:
        print("Using 32 for all the Layers")
        return [32,32,32,32,32]
        
##########   Kernel Size   ############
def get_kernel_size(k_size):
    if type(k_size):
        return [k_size,k_size,k_size,k_size,k_size]
    
def get_optimizer(model , optimizer:str,learning_rate:float):
    if optimizer == "sgd":
        optimizer_fnc = optim.SGD(model.parameters(),
                                 lr=learning_rate, momentum=0.9 ,
                                 , weight_decay=1e-4)
    elif optimizer == "adam":
        optimizer_fnc = optim.Adam(model.parameters(),
                                  lr=learning_rate, weight_decay=1e-4)
    elif optimizer == "rmsprop":
        optimizer_fnc = optim.RMSprop(model.parameters(),
                                     lr=learning_rate,
                                     , weight_decay=1e-4)
    else:
        print("No optimizer Initialized")
    return optimizer_fnc

##########   Dataset loader   ############
def dataset(batch_size):
    
    
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    DATA_DIR = '/kaggle/input/subset-of-original-data/i_nature_sample' 
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    
    full_train_dataset = ImageFolder(root=TRAIN_DIR,transform=transform)

    
    #20 % for validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = math.ceil(0.2 * len(full_train_dataset))
    
    torch.manual_seed(42)
    train_dataset , val_dataset = random_split(
        full_train_dataset,
        [train_size , val_size]
    )
    
    train_loader = DataLoader(train_dataset ,
                              batch_size=batch_size,
                              shuffle=True
                             )
    val_loader = DataLoader(val_dataset ,
                              batch_size=batch_size,
                              shuffle=False
                             )
    return train_loader , val_loader

##########   Training Logic (with wandb agent integrated , train as well as val for each epoch)   ############
def train_test(config=None):
    wandb.init(config=sweep_config)
    config = wandb.config
    
    model = CNNModel(conv_out_channels= get_conv_out_channels(config.conv_out_channels),
                    kernel_size=get_kernel_size(config.kernel_size),
                    dense_layer_out=config.dense_layer_out,
                    activation_func= get_activation_function(config.activation_func),
                    dense_layer_func=get_activation_function(config.dense_layer_func),
                    num_of_class=config.num_of_class,
                    drop_out_input=config.drop_out_input,
                    drop_out_hidden= config.drop_out_hidden)

    print(f"Model initialized with kernel size: {config.kernel_size} and dropout: {config.drop_out_hidden}")
    model.to(device)
    train_loader , val_loader = dataset(config.batch_size)
    optimizer = get_optimizer(model,config.optimizer ,config.learning_rate)

    for epoch in range(config.epochs):
        epoch_loss = train_epoch(model , train_loader , optimizer)
        print(f'Epoch loss:{epoch_loss} ,  Epoch:{epoch+1}')
        val_loss, val_accuracy = validate_epoch(model, val_loader)
        #print(f" Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        wandb.log({"loss": epoch_loss, "epoch": epoch+1,
                  "val_loss": val_loss,"val_accuracy": val_accuracy})
        
##########   Logic For Each Epoch and Back Propagation    ############
def train_epoch(model , train_loader , optimizer):
    model.train()
    running_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx ,(images,labels) in enumerate(train_loader):
        images,labels = images.to(device) , labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs ,labels)

        loss.backward()

        optimizer.step()

        wandb.log({"batch loss": loss.item()})

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss

##########   Logic for Each epoch Validation   ############
def validate_epoch(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    

    # This disables dropout and ensures batch norm uses running statistics
    model.eval()
            
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
        

    with torch.no_grad():
        
        for images, labels in val_loader:
        # Move data to the appropriate device (GPU or CPU)
            images = images.to(device)
            labels = labels.to(device)
            
            # 1. Forward pass
            outputs = model(images)
                        
            # 2. Calculate the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # 3. Calculate accuracy for this batch

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
            # Calculate final metrics for the entire epoch
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = (correct_predictions / total_predictions) * 100 
            

    return epoch_loss, epoch_accuracy
########################    Test Model    

def test_model(model ,batch_size):
    loader = test_loader(batch_size)
    accuracy = test(model ,loader)
    return accuracy

def test_loader(batch_size):
    DATA_DIR = '/kaggle/input/subset-of-original-data/i_nature_sample'
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    test_dataset = ImageFolder(root=TEST_DIR)
    test_loader = DataLoader(test_dataset , 
                            batch_size = batch_size,
                            shuffle = False)
    return test_loader

def test(model , test_loader):
    model.eval()
    correct_pred = 0
    total_pred = 0
    
    with torch.no_grad():
        for images ,labels in test_loader:
            images , labels = images.to(device) , labels.to(device)

            outputs = model(images)
            criterion = nn.CrossEntropyLoss()

            loss = criterion(outputs , labels)
            _ , predicted = torch.max(outputs.data , 1)

            correct_pred += (predicted == labels).sum().item()
            total_pred += labels.size(0)
        
    test_accuracy = 100 * correct_pred /total_pred

    return test_accuracy
        
########################################################
############### Testing Code of Model ##################
'''
TRIED Some Test If model is learning in small dataset and if model's getting right dataset with right class.
Here is Modified train function for This Testing 
'''
def overfit_dataset(batch_size):
    
    
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    DATA_DIR = '/kaggle/input/subset-of-original-data/i_nature_sample' 
    #TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    
    full_train_dataset = ImageFolder(root=TEST_DIR,transform=transform)

    
    #20 % for validation
    #train_size = int(0.8 * len(full_train_dataset))
    #val_size = math.ceil(0.2 * len(full_train_dataset))
    
    #torch.manual_seed(42)
    #train_dataset , val_dataset = random_split(
    #   full_train_dataset,
    #    [train_size , val_size]
    #)
    
    train_loader = DataLoader(full_train_dataset ,
                              batch_size=batch_size,
                              shuffle=True
                             )
    val_loader = DataLoader(full_train_dataset ,
                              batch_size=batch_size,
                              shuffle=False
                             )
    return train_loader , val_loader
    
def train(): #without wandb 
    optimizer = 'adam'
    train_loader , val_loader = overfit_dataset(32)
    optimizer = get_optimizer(model, optimizer ,0.0001)
    
    for i in range(50):
        avg_loss = train_epoch(model , train_loader , optimizer)
        print('epoch_loss:',avg_loss)
        val_loss, val_accuracy = validate_epoch(model, val_loader)
        print(f" Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        #print(f"loss: {avg_loss}, epoch: {i+1},val_loss: {val_loss},val_accuracy: {val_accuracy}")

# %% [code] {"execution":{"iopub.status.busy":"2025-12-12T16:51:24.121615Z","iopub.execute_input":"2025-12-12T16:51:24.122181Z","iopub.status.idle":"2025-12-12T16:57:13.740671Z","shell.execute_reply.started":"2025-12-12T16:51:24.122157Z","shell.execute_reply":"2025-12-12T16:57:13.739955Z"}}
train()

# %% [code] {"execution":{"iopub.status.busy":"2025-12-12T16:23:15.437228Z","iopub.execute_input":"2025-12-12T16:23:15.437865Z","iopub.status.idle":"2025-12-12T16:23:15.455568Z","shell.execute_reply.started":"2025-12-12T16:23:15.437837Z","shell.execute_reply":"2025-12-12T16:23:15.455033Z"}}
model_para = model.parameters()
'''
model = CNNModel(
    conv_out_channels=get_conv_out_channels("same_64") ,
    kernel_size= get_kernel_size(3),
    dense_layer_out=512,
    activation_func=nn.ReLU,
    dense_layer_func=nn.Sigmoid,
    num_of_class=10,
    drop_out_input=0.1,
    drop_out_hidden=0.2
).to(device)
'''
param_dir = "/kaggle/working/model_parameters"
os.makedirs(param_dir, exist_ok=True)
param_path = os.path.join(param_dir, "same_54_k3.pth")

# Save only the parameters (state_dict)
torch.save(model.state_dict(), param_path)

# %% [code] {"execution":{"iopub.status.busy":"2025-12-12T16:46:28.196417Z","iopub.execute_input":"2025-12-12T16:46:28.197388Z","iopub.status.idle":"2025-12-12T16:46:28.216476Z","shell.execute_reply.started":"2025-12-12T16:46:28.197360Z","shell.execute_reply":"2025-12-12T16:46:28.215862Z"}}
#overfit data #small val_data 35% accuracy
torch.save(model.state_dict(),param_path)

# %% [code] {"execution":{"iopub.status.busy":"2025-12-12T16:58:28.433807Z","iopub.execute_input":"2025-12-12T16:58:28.434623Z","iopub.status.idle":"2025-12-12T16:58:28.438020Z","shell.execute_reply.started":"2025-12-12T16:58:28.434596Z","shell.execute_reply":"2025-12-12T16:58:28.437229Z"}}
#one more overfit data (only test data small data to see if dataset function is correct )
#(99% accuracy ) ran it for 50 epochs . removed Horizontal flip from transform 
