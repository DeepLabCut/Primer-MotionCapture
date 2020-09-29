#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses Imgaug to display various augmentation methods
for a few labeled images of a mouse (Data in folder mouse_m7s3
from Mathis, A., et al. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning.
Nat Neurosci 21, 1281–1289 (2018). https://doi.org/10.1038/s41593-018-0209-y)

For "A Primer on Motion Capture with DeepLearning:
Principles, Pitfalls and Perspectives"

by Alexander Mathis, Steffen Schneider,
Jessy Lauer, and Mackenzie Weygandt Mathis

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

imfolder='mouse_m7s3'
Dataframe = pd.read_hdf(os.path.join(imfolder,"CollectedData_Pranav.h5"))

scorer=Dataframe.columns.get_level_values(0)[0]
bodyparts=Dataframe.columns.get_level_values(1)

ia.seed(1)

#parameters for plotting:
color=(200,0,0)
size=17
alpha=.4

#setting up augmentations
Augmentations=[]

augtype='invert'
seq = iaa.Sequential([iaa.Invert(1, per_channel=0.5)])
Augmentations.append([augtype,seq])

augtype='coarsedropout'
seq = iaa.Sequential([iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)])
Augmentations.append([augtype,seq])

augtype='jpegcompression'
seq = iaa.Sequential([iaa.JpegCompression(compression=(70, 99))])
Augmentations.append([augtype,seq])

augtype='motionblur'
seq = iaa.Sequential([iaa.MotionBlur(k=30)])
Augmentations.append([augtype,seq])

augtype='edgedetect'
seq = iaa.Sequential([iaa.EdgeDetect(alpha=(0.8, 1.0))])
Augmentations.append([augtype,seq])

augtype='flipud'
seq = iaa.Sequential([iaa.Flipud(1)])
Augmentations.append([augtype,seq])

augtype='fliplr'
seq = iaa.Sequential([iaa.Fliplr(1)])
Augmentations.append([augtype,seq])


for ind, imname in enumerate(Dataframe.index):
        image=imresize(imread(os.path.join(imfolder,imname)),size=scale)
        ny,nx,nc=np.shape(image)

        kpts=[]
        for b in bodyparts:
            x, y=Dataframe.iloc[ind][scorer][b]['x'], Dataframe.iloc[ind][scorer][b]['y']
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

            cells.append(image_after[:ny,:nx,:nc])

        grid_image = np.hstack(cells)  # Horizontally stack the images
        imageio.imwrite('augmentationexamples/'+str(imfolder)+'_'+imname.split('.png')[0]+'_joint.jpg', grid_image)
