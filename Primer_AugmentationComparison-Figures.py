#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script used plotting Fig 8 a-e without c.

AM, May 29th 2020

For "A Primer on Motion Capture with Deep Learning: Principles, Pitfalls and Perspectives"
by Alexander Mathis, Steffen Schneider, Jessy Lauer, and Mackenzie Weygandt Mathis

"""

import os
import pandas as pd
from pathlib import Path
import deeplabcut
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

params = {
   'axes.labelsize': 22,
   'legend.fontsize': 12,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18,
   'text.usetex': False,
   'figure.figsize': [5,5],
   'font.size': 22,
   'axes.linewidth': 2,
   'xtick.major.size': 5,
   'xtick.major.width': 2,
   'ytick.major.size': 5,
   'ytick.major.width': 2
   }

plt.rcParams.update(params)
def figurchen():
    fig = plt.figure()
    ax = fig.add_subplot(111,position=[0.2,.14, .77,.82])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

method=['imgaug','scalecrop','tensorpack']

cmap = plt.cm.get_cmap('viridis', 4)
colors = cmap(range(4))

##########################################
############# Load the loss & pixel errors
##########################################

figurchen()
for shuffle in [1,2,3]:
    #fn=os.path.join(basepath,'openfield-Pranav-2018-10-30/dlc-models/iteration-0/openfieldOct30-trainset95shuffle'+str(shuffle)+'/train/learning_stats.csv')
    fn='ResultsComparison/learning_stats_shuffle'+str(shuffle)+'.csv'

    lossdata=pd.read_csv(fn)
    iterations,loss,learninrate=np.array(lossdata).T

    plt.plot(iterations,loss,'-',label=method[shuffle-1],alpha=.9,color=colors[shuffle-1])

plt.xticks([0,25000,50000,75000,100000],["0","25k","50k","75k","100k"])
plt.yticks([0, 0.010],["0","0.01"])
plt.xlabel("Training iterations")
plt.ylabel("Training loss")

plt.legend()
plt.savefig('ResultsComparison/LearningCurves.png')

figurchen()
for shuffle in [1,2,3]:
    fn='ResultsComparison/DLC_resnet50_openfieldOct30shuffle'+str(shuffle)+'_100000-results.csv'
    resultsdata = pd.read_csv(fn)
    iterations=resultsdata['Training iterations:']
    trainerror=resultsdata[' Train error(px)']
    testerror=resultsdata[' Test error(px)']

    plt.plot(iterations,trainerror.values,'-',alpha=.2,color=colors[shuffle-1])
    plt.plot(iterations,testerror.values,'.--',alpha=.9,color=colors[shuffle-1],label=method[shuffle-1],)

plt.xlabel("Training iterations")
plt.ylabel("Train/Test error in pixels")
plt.legend()
plt.xticks([0,25000,50000,75000,100000],["0","25k","50k","75k","100k"])
plt.yticks([0,2,4,6],["0","2","4","6"])
plt.ylim(0,7)
plt.savefig('ResultsComparison/ErrorCurves.png')


plt.close("all")


#################################################################################
############# Comparing video predictions (aligned stick figures)
#################################################################################

plt.figure(figsize=(10,5),dpi=600)

N=500
for shuffle in [1,2,3]:
    fn='ResultsComparison/m3v1mp4DLC_resnet50_openfieldOct30shuffle'+str(shuffle)+'_100000.h5'
    Dataframe = pd.read_hdf(fn)
    plt.subplot(3,1,shuffle)
    alphavalue=0.1
    scorer=Dataframe.columns.get_level_values(0)[0]
    bodyparts=set(Dataframe.columns.get_level_values(1))


    Delta=10*np.arange(2330)[:N]
    for bpindex, bp in enumerate(bodyparts):
        DeltaX=-Dataframe[scorer]['tailbase']['x'].values[:N]+Delta
        DeltaY=-Dataframe[scorer]['tailbase']['y'].values[:N]

        for (bp1,bp2) in [('snout','tailbase'),('leftear','rightear')]:
            plt.plot([Dataframe[scorer][bp1]['x'].values[:N]+DeltaX,Dataframe[scorer][bp2]['x'].values[:N]+DeltaX],
                    [Dataframe[scorer][bp1]['y'].values[:N]+DeltaY,Dataframe[scorer][bp2]['y'].values[:N]+DeltaY],color=colors[shuffle-1],alpha=.1)

    plt.axis('off')
plt.savefig('ResultsComparison/Stickfiguresfirst'+str(N)+'.png')

plt.figure(figsize=(10,5),dpi=1200)

for shuffle in [1,2,3]:
    fn='ResultsComparison/m3v1mp4DLC_resnet50_openfieldOct30shuffle'+str(shuffle)+'_100000.h5'
    Dataframe = pd.read_hdf(fn)
    plt.subplot(3,1,shuffle)
    alphavalue=0.1
    scorer=Dataframe.columns.get_level_values(0)[0]
    bodyparts=set(Dataframe.columns.get_level_values(1))


    Delta=10*np.arange(2330)
    for bpindex, bp in enumerate(bodyparts):
        DeltaX=-Dataframe[scorer]['tailbase']['x'].values+Delta
        DeltaY=-Dataframe[scorer]['tailbase']['y'].values

        for (bp1,bp2) in [('snout','tailbase'),('leftear','rightear')]:
            plt.plot([Dataframe[scorer][bp1]['x'].values+DeltaX,Dataframe[scorer][bp2]['x'].values+DeltaX],
                    [Dataframe[scorer][bp1]['y'].values+DeltaY,Dataframe[scorer][bp2]['y'].values+DeltaY],color=colors[shuffle-1],alpha=.1)


    plt.axis('off')

plt.savefig('ResultsComparison/Stickfigures.png')


cap = cv2.VideoCapture('ResultsComparison/m3v1mp4.mp4')

nframes = int(cap.get(7))
strwidth = int(np.ceil(np.log10(nframes)))  # width for strings
ny = int(cap.get(4))
nx = int(cap.get(3))
#plt.figure(figsize=(10,5),dpi=1200)

delta=30
start=85

plt.figure() #figsize=(10,5),dpi=1200)

for shuffle in [1,2,3]:
    fn='ResultsComparison/m3v1mp4DLC_resnet50_openfieldOct30shuffle'+str(shuffle)+'_100000.h5'
    Dataframe = pd.read_hdf(fn)
    plt.clf()
    alphavalue=0.1
    scorer=Dataframe.columns.get_level_values(0)[0]
    bodyparts=set(Dataframe.columns.get_level_values(1))


    for index in tqdm(start+np.arange(0,delta+1,5)): #range(nframes)):
        cap.set(1, index)
        ret, frame = cap.read()
        imname = "frame" + str(index).zfill(strwidth)
        if ret and index==start:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)

        #for (bp1,bp2) in [('snout','tailbase'),('leftear','rightear')]:
        #    plt.plot([Dataframe[scorer][bp1]['x'].values[index],Dataframe[scorer][bp2]['x'].values[index]],
        #                [Dataframe[scorer][bp1]['y'].values[index],Dataframe[scorer][bp2]['y'].values[index]],color=colors[shuffle-1],alpha=.75,lw=3)
        for (bp1,bp2) in [('tailbase','snout',),('leftear','rightear')]:
            plt.arrow(Dataframe[scorer][bp1]['x'].values[index],Dataframe[scorer][bp1]['y'].values[index],
                      Dataframe[scorer][bp2]['x'].values[index]-Dataframe[scorer][bp1]['x'].values[index],
                      Dataframe[scorer][bp2]['y'].values[index]-Dataframe[scorer][bp1]['y'].values[index],color=colors[shuffle-1],alpha=.5,lw=3,head_width=9)

        if index==start+delta:
            image_output = os.path.join('ResultsComparisonFrames', imname + str(method[shuffle-1])+"startframe.png")
            plt.axis("off")
            plt.savefig(image_output)

        if ret and index==start+delta:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
        print(index)
        if index==start+delta:
            image_output = os.path.join('ResultsComparisonFrames', imname + str(method[shuffle-1])+"stopframe.png")
            plt.axis("off")
            plt.savefig(image_output)
