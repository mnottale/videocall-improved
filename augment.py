#! /usr/bin/env python3

""" Augment a dataset by rotating and scaling
"""

import os
import math
import json
import cv2
import dlib
import numpy as np
import sys
import argparse
import datetime
import random

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input dir')
parser.add_argument('-o', '--output', help='output dir')
parser.add_argument('-c', '--count', help='number of images to generate per input')
args = parser.parse_args()

rrange = [-45, 45]
srange = [0.7, 1.2]
rnd = random.Random()

for f in os.listdir(args.input):
    img = cv2.imread('{}/{}'.format(args.input, f))
    w = img.shape[1] // 2
    left = img[:,0:w,:]
    right = img[:,w:w*2,:]
    for k in range(int(args.count)):
        rot = random.randint(rrange[0], rrange[1])
        sc = random.randint(0, 10000) * (srange[1]-srange[0]) / 10000.0 + srange[0]
        image_center = tuple(np.array(left.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rot, sc)
        rleft = cv2.warpAffine(left, rot_mat, left.shape[1::-1], flags=cv2.INTER_LINEAR)
        rright = cv2.warpAffine(right, rot_mat, left.shape[1::-1], flags=cv2.INTER_LINEAR)
        res = np.zeros_like(img)
        res[:,0:w,:] = rleft
        res[:,w:w*2,:] = rright
        cv2.imwrite('{}/{}-{}.png'.format(args.output, f, k), res)