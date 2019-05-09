#Script that tests a trained ResNet Model on a select group of testing data (no patient overlap)
#Loads in a trained model on pneumonia detection
#Prints accuracy output on various batches and takes aggregate average
#Author: Suraj Shah
#Date Last Modified: 5/6/19

#Import statements
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import json
from PIL import Image
from collections import OrderedDict

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

data_dir = 'data/MIMICS/'
TRAIN = 'trainBIG'
VAL = 'valBIG'
TEST = 'valM'

#Transform data on training, validation, test
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally.
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),

    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    ) for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    ) for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

#shows number of images in each set
for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

#shows the number of classes
print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)
print(image_datasets[VAL].classes)
print(image_datasets[TEST].classes)

#Load in model classand the pretrained model
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('resLarge.pth'))
model.eval()

model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#Calculate the accuracy on a test model
def calc_accuracy(model, data, cuda=False):
    model.eval()
    model.to(device='cuda')
    overall_acc = 0
    count = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)

            count = count + 1
            equals = predicted == labels.data
            #if idx == 0:
                #print(equals)
            overall_acc = overall_acc + equals.float().mean()
            print(equals.float().mean())
    return overall_acc / count

#Execute the test
x = calc_accuracy(model, TEST, True)

#Print average accuracy
print(x)

