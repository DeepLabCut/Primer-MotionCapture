#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Evaluting the models on the whole dataset and displaying results.
AM, May 29th 2020

For "A Primer on Motion Capture with Deep Learning: Principles, Pitfalls and Perspectives"
by Alexander Mathis, Steffen Schneider, Jessy Lauer, and Mackenzie Weygandt Mathis

"""


import os
import pandas as pd
import numpy as np
from pathlib import Path

os.environ["DLClight"] = "True"

import deeplabcut

#########################################################
##### Results for evaluation on the large dataset (before finetuning on additional frames)
#########################################################

#Notes: mouse 5 - 8 domain shift (i.e. different camera)
#m4 >> training mouse;
for shuffle in [1, 2, 3]:
    RMSE=pd.read_hdf("ResultsComparison/Errors_shuffle"+str(shuffle)+".h5", "df_with_missing")

    differentmouse_samecamera=[image for image in RMSE.index if  '/m1' in image or '/m2' in image or '/m3' in image]
    differentmouse_differentcamera=[image for image in RMSE.index if  '/m5' in image or '/m6' in image or '/m7' in image or '/m8' in image]

    print(shuffle)
    print(np.mean(RMSE.loc[differentmouse_samecamera]))

error_files = ['ResultsComparison/Errors_shuffle1.h5',
               'ResultsComparison/Errors_shuffle2.h5',
               'ResultsComparison/Errors_shuffle3.h5']

augmenters = ['imgaug', 'scalecrop', 'tensorpack']
data = []
for n, file in enumerate(error_files):
    df = pd.read_hdf(file)
    temp = df.stack().reset_index()
    temp['aug'] = augmenters[n]
    data.append(temp)
data = pd.concat(data).set_index('level_0')
data.columns = ['Bodyparts', 'Error', 'Augmenters']
same_camera = data.index.str.contains('/m1|/m2|/m3')
diff_camera = data.index.str.contains('/m5|/m6|/m7|/m8')

import matplotlib.pyplot as plt
# Produce the PCK curves
data.loc[same_camera, 'Camera'] = 'same'
data.loc[diff_camera, 'Camera'] = 'diff'
bpts = ['snout', 'leftear', 'rightear', 'tailbase']
cmap = plt.cm.get_cmap('viridis', 4)
colors = cmap(range(4))
fig, axes = plt.subplots(2, 4, figsize=(5.32, 3.14), dpi=200)
for (mask, bpt, aug), df in data.groupby(['Camera', 'Bodyparts', 'Augmenters']):
    n = bpts.index(bpt)
    sorted_errors = np.sort(df['Error'])
    n_detections = len(sorted_errors) + 1
    x = np.concatenate([sorted_errors, sorted_errors[[-1]]])
    y = np.linspace(0, 1, n_detections)
    axes[int(mask == 'diff'), n].step(x, y, color=colors[augmenters.index(aug)], lw=2)
for n, ax in enumerate(axes.flat):
    #ax.set_box_aspect(1)
    ax.tick_params(axis='both', direction='in')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if n < 4:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines['bottom'].set_visible(False)
    if n % 4 != 0:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
patches = [plt.plot([0, 0], [0, 0], color=color, label=f'{label}', lw=2)[0]
           for color, label in zip(colors, augmenters)]
fig.legend(handles=patches, loc='center', frameon=False,
           bbox_to_anchor=(0, 0.45, 1, 0.1),
           ncol=3, borderaxespad=0.)

plt.savefig("ResultsComparison/PCKresults.png")

#########################################################
##### Results for evaluation on the large dataset (after finetuning on additional frames from mouse 7)
#########################################################


nameprefix="ResultsComparison/ErrorsafterAugmentation"
error_files = [nameprefix+"_shuffle1.h5",
               nameprefix+"_shuffle2.h5",
               nameprefix+"_shuffle3.h5"]

augmenters = ['imgaug', 'scalecrop', 'tensorpack']
data = []
for n, file in enumerate(error_files):
    df = pd.read_hdf(file)
    temp = df.stack().reset_index()
    temp['aug'] = augmenters[n]
    data.append(temp)
data = pd.concat(data).set_index('level_0')
data.columns = ['Bodyparts', 'Error', 'Augmenters']

same_camera = data.index.str.contains('/m1|/m2|/m3')
diff_camera = data.index.str.contains('/m5|/m6|/m8') #DROPPING mouse 7 as it has now 5 frames in training set!

# Produce the PCK curves
data.loc[same_camera, 'Camera'] = 'same'
data.loc[diff_camera, 'Camera'] = 'diff'
bpts = ['snout', 'leftear', 'rightear', 'tailbase']
cmap = plt.cm.get_cmap('viridis', 4)
colors = cmap(range(4))
fig, axes = plt.subplots(2, 4, figsize=(5.32, 3.14), dpi=200)
for (mask, bpt, aug), df in data.groupby(['Camera', 'Bodyparts', 'Augmenters']):
    n = bpts.index(bpt)
    sorted_errors = np.sort(df['Error'])
    n_detections = len(sorted_errors) + 1
    x = np.concatenate([sorted_errors, sorted_errors[[-1]]])
    y = np.linspace(0, 1, n_detections)
    axes[int(mask == 'diff'), n].step(x, y, color=colors[augmenters.index(aug)], lw=2)
for n, ax in enumerate(axes.flat):
    #ax.set_box_aspect(1)
    ax.tick_params(axis='both', direction='in')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if n < 4:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines['bottom'].set_visible(False)
    if n % 4 != 0:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
patches = [plt.plot([0, 0], [0, 0], color=color, label=f'{label}', lw=2)[0]
           for color, label in zip(colors, augmenters)]
fig.legend(handles=patches, loc='center', frameon=False,
           bbox_to_anchor=(0, 0.45, 1, 0.1),
           ncol=3, borderaxespad=0.)

plt.savefig("ResultsComparison/PCKresultsafterAugmentation.png")
