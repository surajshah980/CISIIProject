#This script creates a data ingestion pipeline into sorted classes that follow
#a tree structure which is enables seamless input for a PyTorch model
#By specifying the data that one wants to reorganize, one can create the copies of the
#data inot a specified workspace
#Used for both MIMICS and NIH datasets
#Author: Suraj Shah
#Date Last Modified: 4/24/19

%matplotlib inline
from IPython.display import display
from IPython.display import Image as _Imgdis
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import ndimage
import torch
import torchvision
from torch import autograd
import skimage
from skimage import transform
from skimage import filters
from skimage.transform import rescale, resize, downscale_local_mean
import PIL
from PIL import Image, ImageOps
import glob
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import copy
import os
import shutil
import pandas as pd #import pandas package and creating alias for namespace

#Create a pandas dataframe from an input CSV and drop the unnecessary columns (the other thoracic pathologies)
df = pd.read_csv("trainMIMICS.csv")
df = df.drop('No Finding', axis=1)
df = df.drop('Enlarged Cardiomediastinum', axis=1)
df = df.drop('Cardiomegaly', axis=1)
df = df.drop('Airspace Opacity', axis=1)
df = df.drop('Lung Lesion', axis=1)
df = df.drop('Edema', axis=1)
df = df.drop('Consolidation', axis=1)
df = df.drop('Atelectasis', axis=1)
df = df.drop('Pneumothorax', axis=1)
df = df.drop('Pleural Effusion', axis=1)
df = df.drop('Pleural Other', axis=1)
df = df.drop('Fracture', axis=1)
df = df.drop('Support Devices', axis=1)
df = df[df.view == 'frontal']
df = df.drop('view', axis=1)

#Specify the number of images left after dropping lateral (scope limited to frontal)
TrainArray = df.values
TrainPathArray = TrainArray[:, 0]
NUM_IMAGES = 248285
#print(TrainArray.shape)

#Readjusting the column labels for sparse inputs
for i in range(NUM_IMAGES):
    if(TrainArray[i, 1] != 1):
        TrainArray[i, 1] = 0

#Print the number of images in our training set
count = 0
for i in range(NUM_IMAGES):
    x = ''.join(TrainArray[i, 0])
    x = 'data/' + x
    if(os.path.isfile(x)):
        count = count + 1
print(count)

#Move the files to specified classes
count = 0
for i in range(NUM_IMAGES):
    x = ''.join(TrainArray[i, 0])
    x = 'data/' + x
    if(os.path.isfile(x)):
        #count = count + 1
        if(TrainArray[i, 1] == 1):
            #move file to 1 folder
            if (count < 65000):
                valid_path = 'data/MIMICS/trainBIG/1/'
            else:
                valid_path = 'data/MIMICS/valBIG/1/'
            temp = x.split('/')
            file = temp[1] + temp[2] + temp[3] + temp[4]
            valid_path = valid_path + file
            #os.rename(x, valid_path)
            shutil.copy(x, valid_path)
        else:
            #move file to 0 folder
            if (count < 65000):
                valid_path = 'data/MIMICS/trainBIG/0/'
            else:
                valid_path = 'data/MIMICS/valBIG/0/'
            temp = x.split('/')
            file = temp[1] + temp[2] + temp[3] + temp[4]
            valid_path = valid_path + file
            #os.rename(x, valid_path)
            shutil.copy(x, valid_path)
        count = count + 1

#Repeat process for testing set
dfV = pd.read_csv("validMIMICS.csv")
dfV = dfV.drop('No Finding', axis=1)
dfV = dfV.drop('Enlarged Cardiomediastinum', axis=1)
dfV = dfV.drop('Cardiomegaly', axis=1)
dfV = dfV.drop('Airspace Opacity', axis=1)
dfV = dfV.drop('Lung Lesion', axis=1)
dfV = dfV.drop('Edema', axis=1)
dfV = dfV.drop('Consolidation', axis=1)
dfV = dfV.drop('Atelectasis', axis=1)
dfV = dfV.drop('Pneumothorax', axis=1)
dfV = dfV.drop('Pleural Effusion', axis=1)
dfV = dfV.drop('Pleural Other', axis=1)
dfV = dfV.drop('Fracture', axis=1)
dfV = dfV.drop('Support Devices', axis=1)
dfV = dfV[dfV.view == 'frontal']
dfV = dfV.drop('view', axis=1)

#Set specific number of images in testing set
ValidArray = dfV.values
ValidPathArray = ValidArray[:, 0]
NUM_VALID_IMAGES = 1759

#Rearrange columns
for i in range(NUM_VALID_IMAGES):
    if(ValidArray[i, 1] != 1):
        ValidArray[i, 1] = 0

#Repeat process 
count = 0
for i in range(NUM_VALID_IMAGES):
    x = ''.join(ValidArray[i, 0])
    x = 'data/' + x
    if(os.path.isfile(x)):
        count = count + 1
        if(ValidArray[i, 1] == 1):
            #move file to 1 folder
            valid_path = 'data/MIMICS/valM/1/'
            temp = x.split('/')
            file = temp[1] + temp[2] + temp[3] + temp[4]
            valid_path = valid_path + file
            #os.rename(x, valid_path)
            shutil.copy(x, valid_path)
        else:
            #move file to 0 folder
            valid_path = 'data/MIMICS/valM/0/'
            temp = x.split('/')
            file = temp[1] + temp[2] + temp[3] + temp[4]
            valid_path = valid_path + file
            #os.rename(x, valid_path)
            shutil.copy(x, valid_path)


