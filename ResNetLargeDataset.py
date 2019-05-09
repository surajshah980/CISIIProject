#Script that returns a trained ResNet model on chest x-ray data for pneumonia detection
#Can specify optimizations and epochs for training
#Outputs the training and validation accuracy for each epoch, as well as the roc_curve
#and AUC (to determine the true acuracy of a binary classifier)
#This specific script implements training the model on the largest training set available (65K images)
#Author: Suraj Shah
#Date Last Modified: 5/7/19
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
import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#make sure that we are taking full advantage of the GPU capabilities of MARCC
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

#Training on the large dataset
data_dir = 'data/MIMICS/'
TRAIN = 'trainBIG'
VAL = 'valBIG'

# ResNet Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we downscale crop the image to 224x224 and
        # randomly flip it horizontally.
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
}

#Set up transfer links to Image Folder
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    ) for x in [TRAIN, VAL]
}

#DataLoader is the preferred ingestion pipeline for Pytorch models
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    ) for x in [TRAIN, VAL]
}

#Specify dataset size (number of images)
dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}

#shows number of images in each set
for x in [TRAIN, VAL]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

#shows the number of classes
print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)
print(image_datasets[VAL].classes)

#Load the pre-trained model and set number of output features (reduce from 1000 to 2)
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

#Specify the parameters (BinaryCrossEntropy Loss, Stochastic Gradient Descent, etc.)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
eps=10 #specify number of epochs

#Function that implements training the model
def train_model(model, criteria, optimizer, scheduler,
                                      num_epochs=10):

    #Record overall time it takes to train and validate
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [TRAIN, VAL]:
            if phase == TRAIN:
                scheduler.step()
                model.train()  #Set model to training mode
            else:
                model.eval()   #Set model to evaluate mode

            #Set the loss and number of correct prediction
            running_loss = 0.0
            running_corrects = 0
            new_pred = []
            new_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == TRAIN):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()

                        # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                #Append to list for input into ROC curve
                for i in range(len(preds.cpu().numpy())):
                    new_pred.append(preds.cpu().numpy()[i])
                for i in range(len(labels.cpu().numpy())):
                    new_labels.append(labels.cpu().numpy()[i])



            new_labels = np.array(new_labels)
            new_pred = np.array(new_pred)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #Generate the false positive and true positive rate
            fpr, tpr, _ = sklearn.metrics.roc_curve(new_labels, new_pred)
            roc = sklearn.metrics.auc(fpr, tpr)

            #Plot the figure
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.show()

            #Print the accuracy and loss for each epoch
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('AUC: {:.4f}'.format(roc))

            # deep copy the model
            if phase == VAL and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#Execute function to train the model
model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler , eps)

#Save trained model
torch.save(model.state_dict(), 'resLarge.pth')