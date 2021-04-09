#! /usr/bin/env python3

""" Add a background texture to images with a transparent background
"""

import sys
import os
import numpy as np
import cv2
import random

inputdir = sys.argv[1]
outputdir = sys.argv[2]
texture = sys.argv[3]

rnd = random.Random()
texture = cv2.imread(texture, cv2.IMREAD_COLOR)

for filename in os.listdir(inputdir):
    imgin = cv2.imread(os.path.join(inputdir, filename), cv2.IMREAD_UNCHANGED)
    imgin = cv2.resize(imgin, (128, 128))
    dy = rnd.randint(0, texture.shape[0]-129)
    dx = rnd.randint(0, texture.shape[1]-129)
    texzone = texture[dy:dy+128,dx:dx+128,:].copy()
    texzone[:,:,0] = texzone[:,:,0] * ((255 - imgin[:,:,3]) / 255)
    texzone[:,:,1] = texzone[:,:,1] * ((255 - imgin[:,:,3]) / 255)
    texzone[:,:,2] = texzone[:,:,2] * ((255 - imgin[:,:,3]) / 255)
    imgin = imgin [:,:,0:3] + texzone[:,:,0:3]
    cv2.imwrite(os.path.join(outputdir, filename), imgin)