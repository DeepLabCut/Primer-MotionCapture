#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses Imgaug to display various augmentation methods for a few labeled images of a mouse
(Data recorded with Alex' iphone at Aiguille du Midi near Mont Blanc)

For "A Primer on Motion Capture with Deep Learning: Principles, Pitfalls and Perspectives"
by Alexander Mathis, Steffen Schneider, Jessy Lauer, and Mackenzie Weygandt Mathis

Uses Imgaug:
Code: https://github.com/aleju/imgaug
Docs: https://imgaug.readthedocs.io/en/latest/index.html
"""

import pandas as pd
import os
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

import imageio
from deeplabcut.utils.auxfun_videos import imread, imresize

scale=.4
##########################
## Loading data
##########################
imfolder='montblanc_images'
Dataframe = pd.read_hdf(os.path.join(imfolder,"CollectedData_Daniel.h5"))

scorer=Dataframe.columns.get_level_values(0)[0]
individuals=Dataframe.columns.get_level_values(1)
bodyparts=Dataframe.columns.get_level_values(2)

ia.seed(1)

#parameters for plotting:
color=(200,0,0)
size=13
alpha=.15

#setting up augmentations
Augmentations=[]

augtype='rotateandscale'
#rotate & scale
seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
    iaa.Affine(
        rotate=23,
        scale=(0.9, 1.1)
    ) # rotate by exactly 23 deg and scale to 90-10%, affects keypoints
])
Augmentations.append([augtype, seq])

augtype='fog'
seq = iaa.Sequential([iaa.Fog()])
Augmentations.append([augtype,seq])

augtype='snow'
seq = iaa.Sequential([iaa.Snowflakes(flake_size=(.2,.5),density=(0.005, 0.07), speed=(0.01, 0.05))])
Augmentations.append([augtype,seq])


for ind, imname in enumerate(Dataframe.index):
        image=imresize(imread(os.path.join('montblanc_images',imname)),size=scale)
        ny,nx,nc=np.shape(image)

        kpts=[]
        for i in individuals:
            for b in bodyparts:
                x, y=Dataframe.iloc[ind][scorer][i][b]['x'], Dataframe.iloc[ind][scorer][i][b]['y']
                if np.isfinite(x) and np.isfinite(y):
                    kpts.append(Keypoint(x=x*scale,y=y*scale))

        kps=KeypointsOnImage(kpts, shape=image.shape)

        cells=[]

        # image with keypoints before augmentation
        image_before = kps.draw_on_image(image, color=color,size=size,alpha=alpha)
        cells.append(image_before)

        for name, seq in Augmentations:
            image_aug, kps_aug = seq(image=image, keypoints=kps)
            image_after = kps_aug.draw_on_image(image_aug, color=color,size=size,alpha=alpha)
            cells.append(image_after)

        grid_image = np.hstack(cells)  # Horizontally stack the images
        imageio.imwrite('augmentationexamples/'+str(imfolder)+'_'+imname.split('.png')[0]+'_joint.jpg', grid_image)
