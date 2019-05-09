#Implements Class Activation Mapping Method by heatmapping the feature outputs
#from the last weighted layer of the model
#Improves upon method developed by CSAIL at MIT, generates prediction from
#Trained model on own datasets
#Author: Suraj Shah
#Date Last Modified: 5/7/19

import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import os
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict
use_gpu = torch.cuda.is_available()

#make sure that we are taking full advantage of the GPU capabilities of MARCC
if use_gpu:
    print("Using CUDA")

#Specify the data directories (training and validation that we're working with)
data_dir = 'data/MIMICS/'
TRAIN = 'trainM'
VAL = 'valM'

#Normalize the data for ingestion into the models
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we downscale the image to 224x224 and
        # randomly flip it horizontally.
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #Normalized based off mean and standard deviation of ImageNet
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #Normalized based off mean and standard deviation of ImageNet
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

#Specify the datasets and load in for ingestion to Pytorch Model
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    ) for x in [TRAIN, VAL]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    ) for x in [TRAIN, VAL]
}

#Load the model class
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
#Load in weights andd state of trained model
model.load_state_dict(torch.load('resLarge.pth'))
model.eval()

#Set the final convolutional layer
finalconv_name = 'layer4'

#Hook the features of the model
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

#Implement hook feature on model
model._modules.get(finalconv_name).register_forward_hook(hook_feature)

#Create a weighted softmax (final activation layer) from the model's parameters
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

#returnCAM function returns the outputted class activation image after realizing
#the weightes of the final layer and visualizing them
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

#Specify the image that want to produce class activation for
data = 'data/MIMICS/trainM/1/trainp10012498s01view1_frontal.jpg'
image_pil = Image.open(test)

#Create Normalization and Preprocessing functions for test image
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
   normalize
])

#Finally output the prediction for the image from the model
img_tensor = preprocess(image_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = model(img_variable)

classes = image_datasets[TRAIN].classes
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

#Take the top prediction from the number of classes
for i in range(0, 2):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

#REturn class activation for our input
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

#Apply the color map to the image input
img = cv2.imread('data/MIMICS/trainM/1/trainp10012498s01view1_frontal.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5

#Write to workspace
cv2.imwrite('outputCAM.jpg', result)


