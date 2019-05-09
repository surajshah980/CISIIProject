#This script creates a data ingestion pipeline into sorted classes that follow
#a tree structure which is enables seamless input for a PyTorch model
#By specifying the data that one wants to reorganize, one can create the copies of the
#data inot a specified workspace
#This specific script splits the images into three classes, No_Finding, Other, and Pneumonia
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

df = pd.read_csv("trainMIMICS.csv")
#df = df.drop(df['view' == 'lateral'].index)
df = df[df.view == 'frontal']
print(df.shape)

TrainArray = df.values
TrainPathArray = TrainArray[:, 0]
NUM_IMAGES = 248285
#print(TrainArray.shape)

#Count the numer of training files available
count = 0
for i in range(NUM_IMAGES):
    x = ''.join(TrainArray[i, 0])
    x = 'data/' + x
    if(os.path.isfile(x)):
        count = count + 1
print(count)

count = 0
for i in range(NUM_IMAGES):
    x = ''.join(TrainArray[i, 0])
    x = 'data/' + x
    if(os.path.isfile(x)):
        if(TrainArray[i, 2] == 1):
            #move file to 1 folder
            if (count < 65000):
                valid_path = 'data/MIMICS/train3BIG/No_Finding/'
            else:
                valid_path = 'data/MIMICS/val3BIG/No_Finding/'
            temp = x.split('/')
            file = temp[1] + temp[2] + temp[3] + temp[4]
            valid_path = valid_path + file
            #os.rename(x, valid_path)
            shutil.copy(x, valid_path)
        elif (TrainArray[i, 9] == 1):
            #move file to 0 folder
            if (count < 65000):
                valid_path = 'data/MIMICS/train3BIG/Pneumonia/'
            else:
                valid_path = 'data/MIMICS/val3BIG/Pneumonia/'
            temp = x.split('/')
            file = temp[1] + temp[2] + temp[3] + temp[4]
            valid_path = valid_path + file
            #os.rename(x, valid_path)
            shutil.copy(x, valid_path)
        else:
            if (count < 65000):
                valid_path = 'data/MIMICS/train3BIG/Other/'
            else:
                valid_path = 'data/MIMICS/val3BIG/Other/'
            temp = x.split('/')
            file = temp[1] + temp[2] + temp[3] + temp[4]
            valid_path = valid_path + file
            #os.rename(x, valid_path)
            shutil.copy(x, valid_path)
        count = count + 1