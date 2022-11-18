import numpy as np 
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
from dataloader import LoadDataSet
from utils import EarlyStopping, LRScheduler
from network import Encoder
from sklearn.model_selection import KFold    

torch.backends.cudnn.benchmark = True

batch_size = 1000
nepochs = 200
lr = 0.0001
#train_data_dir = "datasets/step_0.01_isovalue_0.5/width_range_0.5_step_0.01_2M_2.txt"
#test_data_dir = "datasets/step_0.01_isovalue_0.5/width_range_0.5_step_0.01_200k_2.txt"
pretrained_model_dir = ""
data_dir = "datasets/three_wind_datasets/training_data.npy"
# data_dir = "datasets/gaussian/training_sfc_temperature.npy"
# data_dir = "datasets/gaussian/training_wind_pressure.npy"
# data_dir = "datasets/gaussian/training_redsea.npy"
start_epoch = 0
use_lr_scheduler = True
k_folds = 5


savefolder = "three_wind_datasets"

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
checkpoint_dir = os.path.join("checkpoints")

# if not os.path.exists(os.path.join("checkpoints", dataset)):
    # os.mkdir(os.path.join("checkpoints", dataset))
# checkpoint_dir = os.path.join("checkpoints", dataset)

if not os.path.exists(os.path.join(checkpoint_dir, savefolder)):
    os.mkdir(os.path.join(checkpoint_dir, savefolder))
checkpoint_dir = os.path.join(checkpoint_dir, savefolder)

##! set seed 999
manualSeed = 999
torch.manual_seed(manualSeed)

##! device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

##! load datasets
#train_dataset = LoadDataSet(train_data_dir)
#test_dataset = LoadDataSet(test_data_dir)
dataset = LoadDataSet(data_dir)
# train_size = int(len(dataset) * 0.9)
# test_size = int(len(dataset) - train_size)
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#dataset = ConcatDataset([train_dataset, test_dataset])
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)

kfold = KFold(n_splits=k_folds, shuffle=True)

#! Model initialize
# if activation == "identity":
    # activation_func = nn.Identity()
# elif activation == "ReLU":
    # activation_func = nn.ReLU()
# elif activation == "sigmoid":
    # activation_func = nn.Sigmoid()
# elif activation == "Tanh":
    # activation_func = nn.Tanh()
# elif activation == "sine":
    # activation_func = None
# else:
    # print("Activation Function Error")

# gaussian_model = GaussianFourierFeatureTransform(8, 128)

# model = SirenNet(
    # dim_in = 14,                        # input dimension, ex. 2d coor
    # dim_hidden = num_hidden,                  # hidden dimension
    # dim_out = 1,                       # output dimension, ex. rgb value
    # num_layers = num_layers,                    # number of layers
    # final_activation = activation_func,   # activation of final layer (nn.Identity() for direct output)
    # w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
# )
model = Encoder(activation=None)
if pretrained_model_dir != "":
    model.load_state_dict(torch.load(pretrained_model_dir))

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    # gaussian_model = torch.nn.DataParallel(gaussian_model)

model = model.to(device)
# gaussian_model = gaussian_model.to(device)

L2_loss = nn.MSELoss()
L1_loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-06)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# writer = SummaryWriter()

if use_lr_scheduler:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

train_loss = []
test_loss =[]

start_time.record()

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=16, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler, num_workers=16, pin_memory=True)

    for epoch in range(start_epoch, nepochs):
        avg_train_loss = 0
        train_bar = tqdm(enumerate(train_dataloader))
        model.train()
        for i, data in train_bar:
            m = data[0].to(device)
            v = data[1].to(device)
            iso = data[2].to(device)
            target = data[3].to(device)
            model.zero_grad()
            pred = model(m, v, iso)
            # print("target pred", target[0, :], pred[0, :])
            loss = L2_loss(pred, target)
            loss.backward()
            optimizer.step()
            avg_train_loss = avg_train_loss + loss.item()
            # train_bar.set_description(desc= '[%d/%d] Train loss: %.4f' %(epoch+ 1, nepochs, loss.item()))
            # writer.add_scalar('Loss/train', loss, epoch)
        train_loss.append(avg_train_loss / len(train_dataloader))
        print("Average Train Loss:", epoch, avg_train_loss / len(train_dataloader))
        avg_test_loss = 0
        if (epoch + 1) % 1 == 0:
            model.eval()
            test_bar = tqdm(enumerate(test_dataloader))
            for j, test_data in test_bar:
                m = test_data[0].to(device, non_blocking=True)
                v = test_data[1].to(device)
                iso = test_data[2].to(device)
                # mv = gaussian_model(mv)
                target = test_data[2].to(device, non_blocking=True)
                pred = model(m, v, iso)
                loss = L2_loss(pred, target)
                # test_bar.set_description(desc= '[%d/%d] Test loss: %.4f' %(epoch+ 1, nepochs, loss.item()))
                avg_test_loss = avg_test_loss + loss.item()
                # writer.add_scalar('Loss/test', loss, epoch)
            test_loss.append(avg_test_loss / len(test_dataloader))
            print("Average Test Loss: ", epoch, avg_test_loss / len(test_dataloader))
        if use_lr_scheduler == True:
            lr_scheduler(avg_test_loss / len(test_dataloader))

        if (epoch + 1) % 10 == 0:
            # save model 
            path = os.path.join(checkpoint_dir, "model_" + str(epoch+1) + ".pth")
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)
       
            
end_time.record()
torch.cuda.synchronize()
train_path = os.path.join(checkpoint_dir, "train_loss.npy")
np.save(train_path, train_loss)
test_path = os.path.join(checkpoint_dir, "test_loss.npy")
np.save(test_path, test_loss)


print("training time: ", start_time.elapsed_time(end_time) / (1000 * 60 * 60))
