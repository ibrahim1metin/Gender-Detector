import numpy as np
from PIL import Image
import os
from utils import meanOfPairs
trainM=os.listdir("Training/male")
trainF=os.listdir("Training/female")
ValM=os.listdir("Validation/male")
ValF=os.listdir("Validation/female")
sizesTrainM=[]
sizesTrainF=[]
sizesValM=[]
sizesValF=[]
for image in trainM:
    img=Image.open(f"Training/male/{image}")
    sizesTrainM.append(img.size)
for image in trainF:
    img=Image.open(f"Training/female/{image}")
    sizesTrainF.append(img.size)
for image in ValM:
    img=Image.open(f"Validation/male/{image}")
    sizesValM.append(img.size)
for image in ValF:
    img=Image.open(f"Validation/female/{image}")
    sizesValF.append(img.size)
MeanMT=meanOfPairs(sizesTrainF)
MeanFT=meanOfPairs(sizesTrainF)
MeanMV=meanOfPairs(sizesValM)
MeanFV=meanOfPairs(sizesValF)
print(MeanMT)
print(MeanFT)
print(MeanMV)
print(MeanFV)
# the size all images are going to be resized to is (83,108)