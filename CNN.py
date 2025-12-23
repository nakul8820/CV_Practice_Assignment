!pip install wandb 
!pip install lightning

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
import gc


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
            'values': ['same_128', 'double_from_64','double_from_128'] #'same_64',
        },
        'kernel_size': {
            # Allows sweeping a single kernel size used across all layers (e.g. all 3x3 or all 5x5)
            'values': [5 , 7] 
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
            'values': [64] #causig memory crash
        },
        
        'num_of_class': {
            'value': 10 
        },
        'epochs': {
            'value': 10
        }
    }
}


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

conv_out_channel_type = [ 'same_64', 'same_128', 'double_from_64','double_from_128']
def get_conv_out_channels(value:str):
    if value == 'double_from_128':
        return [128,256,512,1024,2048]
    elif value == 'same_64':
        return [64,64,64,64,64]
    elif value == 'same_128':
        return [128,128,128,128,128]
    elif value == 'double_from_64':
        return [64,128,256,512,1024]
    else:
        print("Using 64 for all the Layers")
        return [64,64,64,64,64]        
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
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomRotation(degrees=15),
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
from pytorch_lightning.callbacks import ModelCheckpoint

def train_test():
    # 1. Initialize W&B run
    run = wandb.init(reinit=True)
    wandb_logger = WandbLogger(experiment=run, log_model="all")
    
    # Get config from W&B
    config = run.config
    
    # Setup Data
    train_loader, val_loader = dataset(config.batch_size)
    
    # Initialize Model
    model = LightningCNN(dict(config))
    
    # 2. Define Checkpoint Callback
    # Using run.dir ensures the .ckpt is uploaded to the 'Files' tab automatically
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run.dir, "checkpoints"), 
        monitor='val_accuracy',
        mode='max',
        save_top_k=2,
        filename='best-model-{epoch:02d}-{val_accuracy:.2f}',
        auto_insert_metric_name=False # Makes the filename cleaner
    )
    
    # 3. Setup Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    
    # 4. Train
    trainer.fit(model, train_loader, val_loader)
    
    # 5. Log the best model path & score to the W&B summary for easy filtering
    run.summary["best_model_path"] = checkpoint_callback.best_model_path
    if checkpoint_callback.best_model_score is not None:
        run.summary["best_val_acc"] = checkpoint_callback.best_model_score.item()
    # 6. Cleanup to prevent OOM (Out of Memory) in long sweeps
    run.finish() # Finish first to ensure files are synced
    
    del model, trainer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

best_model = LightningCNN(dict(wandb.config)) 


final_trainer = pl.Trainer(accelerator="gpu", devices=1)
final_trainer.test(best_model, dataloaders=test_loader(64))

########################    Test Model    

def test_loader(batch_size):
    DATA_DIR = '/kaggle/input/subset-of-original-data/i_nature_sample'
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    test_dataset = ImageFolder(root=TEST_DIR)
    test_loader = DataLoader(test_dataset , 
                            batch_size = batch_size,
                            shuffle = False)
    return test_loader
##########################
   #      sweep   # 
sweep_id = wandb.sweep(sweep_config, project="CV_project")

wandb.agent(sweep_id, train_test, count=10)
#########################
  #     Test     #

import wandb
import os
import glob

api = wandb.Api()

# 1. Get the Best Run
sweep = api.sweep(f"nakupatel-indus-university/CV_project/{sweep_id}")
best_run = sweep.best_run()
print(f"Fetching Best Run: {best_run.id}")

# 2. Access the Artifact directly # best of sweep configuration are being downloaded and loaded
artifact_path = f"nakupatel-indus-university/CV_project/model-{best_run.id}:best"


print(f"Downloading artifact: {artifact_path}")
artifact = api.artifact(artifact_path)
download_path = artifact.download()
    
# 3. Find the file in the downloaded folder
ckpt_files = glob.glob(os.path.join(download_path, "**/*.ckpt"), recursive=True)
    
if not ckpt_files:
    raise FileNotFoundError(f"Artifact downloaded to {download_path} but no .ckpt found.")
    
model_path = ckpt_files[0]
print(f"Successfully located checkpoint: {model_path}")

best_model = LightningCNN.load_from_checkpoint(model_path)
trainer = pl.Trainer(accelerator="auto",logger=False, devices=1)
test_results = trainer.test(model=best_model, dataloaders=test_loader(64))
raw_acc = test_results[0]['test_acc']
acc_percent = raw_acc * 100
    
print(f"Final Test Accuracy: {acc_percent:.2f}%")
print(test_results)
