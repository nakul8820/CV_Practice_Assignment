pip install wandb 

import wandb
wand_api_key = ""
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
        'name': 'val_accuracy', # Track a specific metric name from your training loop
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
            'values': ["GELU", "SiLU"] 
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
            'values': [0.1, 0.05]
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
            'values': [64,128]
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

########## Activation Function ########

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

########## Conv Out Channels ########

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

##########   Optimizer   ############
    
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
                              num_workers=3, 
                              pin_memory=True
                             )
    val_loader = DataLoader(val_dataset ,
                              batch_size=batch_size,
                              shuffle=False
                              num_workers=3, 
                              pin_memory=True
                             )
    return train_loader , val_loader

###################### # integrting Pytorch Lightning 
import pytorch_lightning as pl
from torchmetrics import Accuracy

class LightningCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

    #architecture 
        self.conv_layers = nn.Sequential()
        in_channels = 3

        conv_channels = get_conv_out_channels(self.hparams.conv_out_channels)
        kernels = get_kernel_size(self.hparams.kernel_size)
        act_fn = get_activation_function(self.hparams.activation_func)

        self.conv_layers.add_module("drop_out_input" , nn.Dropout(p=self.hparams.drop_out_input))

        for i ,(k_size , out_ch) in enumerate(zip(kernels , conv_channels)):
            self.conv_layers.add_module(f'conv{i+1}', nn.Conv2d(in_channels , out_ch , padding="same"))
            self.conv_layers.add_module(f'bn{i+1}', nn.LazyBatchNorm2d())
            self.conv_layers.add_module(f'activ{i+1}' , act_fn())
            self.conv_layers.add_module(f'piil{i+1}',nn.MaxPool2d(2,2))
            self.conv_layers.add_module(f'dropout{i+1}',nn.Dropout2d(p=self.hparams.drop_out_hidden))
            in_channels = out_ch

        self.dense_layers = nn.Sequential(
            nn.LazyLinear(out_feature = self.hparams.dense_layer_out),
            get_activation_function(self.hparams.dense_layer_func)() ,
            nn.Linear(self.hparams.dense_layer_out , self.hparams.num_of_class)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task = "multiclass" , num_classes=10)
        self.val_acc = Accuracy(task = "multiclass", num_classes=10)

    def forward(self,x):
        x = self.conv_layers(x)
        x = torch.flatten(x,1)
        return self.dense_layers(x)

    def training_step(self,batch , batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits , y)
        self.training_acc(logits,y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy",self.val_acc, prog_bar=True)

    def configure_optimizer(self):
        return get_optimizer(self , self.hparams.optimizer, self.hparams.learning_rate)

def train_test():
    run = wand_test()
    wandb_logger = WandbLogger()

    config = run.config

    train_loader , val_loader = dataset(config.batch_size)

    model = LightningCNN(dict(config))

    #setup trainer for pytorch lightning
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu",
        devices = 1 ,
        precision="16-mixed",
        logger=wandb_logger,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader, val_loader)

    #manual_clean_up
    import gv
    del model,trainer,train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

best_model = LightningCNN(dict(wandb.config)) 


final_trainer = pl.Trainer(accelerator="gpu", devices=1)
final_trainer.test(best_model, dataloaders=test_loader(64))

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
        
