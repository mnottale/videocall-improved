#! /usr/bin/env python3

""" Grab images from a video stream every 300 ms
"""

import os
import math
import json
import cv2
import dlib
import numpy as np
import sys
import argparse
import time
import random


target = sys.argv[1]
count = int(sys.argv[2])

index = 0

width = 640
height = 480
hw = (width-height) // 2
prev_time = time.clock_gettime(0)

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while True:
    _, frame = cap.read()
    now = time.clock_gettime(0)
    zone = frame[:,hw:-hw,:]
    if now - prev_time > 0.3:
        cv2.imwrite('{}/grab-{}.png'.format(target, index), zone)
        prev_time = now
        index = index + 1
        if index >= count:
            break
    cv2.imshow(winname='Feed', mat=zone)
    cv2.waitKey(delay=1)
