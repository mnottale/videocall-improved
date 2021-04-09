#! /usr/bin/env python3

""" Grab a video stream and generate a colored face mask for pix2pix test
    Save both and both side by side when you press space
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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

output_index=0
white = (255, 255, 255)

while True:
    _, frame = cap.read()
    hdw = (640-480) // 2
    frame = frame[:,hdw:640-hdw,:]
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) != 0:
        landmarks = predictor(image=gray, box=faces[0])
        def lm(idx):
            return landmarks.part(idx).x, landmarks.part(idx).y
        oi = np.zeros((480, 480, 3), dtype=np.uint8)
        idxs = list(range(0, 17)) + list(range(26, 16, -1))
        pts = list()
        for i in idxs:
            p = landmarks.part(i)
            pts.append([p.x, p.y])
        ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(oi, [ctr], 0, white, cv2.FILLED)
        pts = list()
        idxs = list(range(36, 42))
        for i in idxs:
            p = landmarks.part(i)
            pts.append([p.x, p.y])
        ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(oi, [ctr], 0, (0, 255, 0), cv2.FILLED)
        pts = list()
        idxs = list(range(42, 48))
        for i in idxs:
            p = landmarks.part(i)
            pts.append([p.x, p.y])
        ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(oi, [ctr], 0, (0, 255, 0), cv2.FILLED)
        pts = list()
        idxs = list(range(49, 60))
        for i in idxs:
            p = landmarks.part(i)
            pts.append([p.x, p.y])
        ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(oi, [ctr], 0, (0, 0, 255), cv2.FILLED)
        seqs = [[0,16], [17, 26], [27, 30], [31, 35], [36, 41], [42, 47], [48, 59]]
        extras = [[0,17], [16, 26], [30, 33], [36, 41], [42, 47], [48, 59]]
        for s in seqs:
            pbegin = s[0]
            pend = s[1]
            while pbegin < pend:
                cv2.line(oi, lm(pbegin), lm(pbegin+1), (255, 0, 0), 2)
                pbegin = pbegin + 1
        for s in extras:
            cv2.line(oi, lm(s[0]), lm(s[1]), (255,0, 0), 2)
        cv2.imshow(winname="out", mat=oi)
    k = cv2.waitKey(delay = 1)
    if k == 27:
        break
    if k == 32:
        cv2.imwrite('gen/in/{}.png'.format(output_index), frame)
        cv2.imwrite('gen/out/{}.png'.format(output_index), oi)
        res = np.zeros((480, 480*2, 3))
        res[:,0:480,:] = frame
        res[:,480:960,:] = oi
        cv2.imwrite('gen/{}.jpg'.format(output_index), res)
        output_index = output_index + 1