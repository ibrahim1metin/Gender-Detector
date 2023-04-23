#imports
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from functools import reduce
from utils import concatLists
#Getting the data
trainM=len(os.listdir("Training/male"))
trainF=len(os.listdir("Training/female"))
ValM=len(os.listdir("Validation/male"))
ValF=len(os.listdir("Validation/female"))
dataTrain=[trainM,trainF]
dataVal=[ValM,ValF]
dataTotal=[reduce(lambda x,y:x+y,dataTrain),reduce(lambda x,y:x+y,dataVal)]
TotalDataGrouped=concatLists(dataTrain,dataVal)
datatrain=np.arange(len(dataTrain))
dataval=np.arange(len(dataVal))

#creating the plot
fig,axs=plt.subplots(2,2,figsize=(10,10))
axs[0,0].set_title("Train Data")
axs[0,1].set_title("Validation Data")
axs[1,0].set_title("Total Data")
axs[1,1].set_title("Data Groups")
axs[0,0].bar(["Male","Female"],[trainM,trainF],color="green")
axs[0,1].bar(["Male","Female"],[ValM,ValF])
for i in range(len(dataTrain)):
    axs[0,0].text(i,dataTrain[i]//2,dataTrain[i],ha="center")
for i in range(len(dataVal)):
    axs[0,1].text(i,dataVal[i]//2,dataVal[i],ha="center")
axs[0,0].bar(datatrain[dataTrain.index(min(dataTrain))], int(max(dataTrain) - min(dataTrain)), bottom=min(dataTrain),color="red")
axs[0,1].bar(dataval[dataVal.index(min(dataVal))], int(max(dataVal) - min(dataVal)), bottom=min(dataVal),color="red")
axs[0,0].text(datatrain[dataTrain.index(min(dataTrain))],
         max(dataTrain)*1.01,
         s=(max(dataTrain) - min(dataTrain)),
         ha="center")
axs[0,1].text(dataval[dataVal.index(min(dataVal))],
         max(dataVal)*1.01,
         s=int(max(dataVal) - min(dataVal)),
         ha="center")
handles = [plt.Rectangle((0, 0), 1, 1, color='green'),plt.Rectangle((0, 0), 1, 1), plt.Rectangle((0, 0), 1, 1, color='red')]
labels = ["The Amount For Traing","The Amount For Validation", "The Difference"]
axs[0,1].legend(handles, labels, bbox_to_anchor=(1, 1),fontsize=7)
colors=sns.color_palette('bright')
axs[1,0].pie(dataTotal,labels=["Train","Validation"],colors=colors,autopct='%.0f%%')
axs[1,1].pie(TotalDataGrouped,labels=["Male Train","Female Train","Male Validation","Female Validation"],colors=colors,autopct='%.0d%%')
plt.show()