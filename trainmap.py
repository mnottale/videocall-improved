#! /usr/bin/env python3

""" pix2pix dataset creation helper. Overlays each image from B in turn with
    video stream so the user can align them.
    Press space to save and move to the next image.
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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input device number')
parser.add_argument('-o', '--output', help='output dir')
parser.add_argument('-m', '--model-dir', help='model with target faces')
args = parser.parse_args()
width = 640
height = 480
offset = (width - height) // 2

cap = cv2.VideoCapture(int(args.input))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

models = os.listdir(args.model_dir)
model_index = 0

def transparentOverlay(src , overlay , pos=(0,0)):
    #print("p {}  os {}  ss {}".format(pos, overlay.shape, src.shape))
    if pos[0] >= src.shape[1] or pos[1] >= src.shape[0] or pos[0]+overlay.shape[1] <= 0 or pos[1]+overlay.shape[0] <= 0:
        return
    # crop check
    if pos[0] + overlay.shape[1] >= src.shape[1]:
        overlay = overlay[:,:src.shape[1]-pos[0], :]
    if pos[1] + overlay.shape[0] >= src.shape[0]:
        overlay = overlay[:src.shape[0]-pos[1],:,:]
    if pos[0] < 0:
        overlay = overlay[-pos[0]:,:,:]
        pos = (0, pos[1])
    if pos[1] < 0:
        overlay = overlay[:,-pos[1]:,:]
        pos = (pos[0], 0)
    zone=src[pos[1]:pos[1]+overlay.shape[0],pos[0]:pos[0]+overlay.shape[1]]
    alpha = overlay[:,:,3] / 255.0
    alpha3 = np.zeros(zone.shape, dtype=np.float64)
    alpha3[:,:,0] = alpha
    alpha3[:,:,1] = alpha
    alpha3[:,:,2] = alpha
    res = zone * (1.0 - alpha3) + overlay[:,:,:3]*alpha3
    src[pos[1]:pos[1]+overlay.shape[0],pos[0]:pos[0]+overlay.shape[1]] = res

def get_model():
    img = cv2.imread('{}/{}'.format(args.model_dir, models[model_index]), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (480, 480))
    img[:, :, 3] = img[:, :, 3] / 2
    return img

model_img = get_model()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) != 0:
        l = predictor(image=gray, box=faces[0])
        idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        pts = list()
        for i in idxs:
            p = l.part(i)
            pts.append([p.x, p.y])
        # add symetrical points from top-nose to get the forehead
        mid = l.part(27)
        for i in idxs:
            p = l.part(i)
            nx = mid.x + mid.x - p.x
            ny = mid.y + mid.y - p.y
            pts.append([nx, ny])
        ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
        msk = np.zeros_like(gray)
        cv2.drawContours(msk, [ctr], 0, 255, cv2.FILLED)
        bmh = cv2.blur(msk, (15, 15))
        alpha3 = np.zeros(frame.shape, dtype=np.float64)
        alpha3[:,:,0] = bmh / 255.0
        alpha3[:,:,1] = bmh / 255.0
        alpha3[:,:,2] = bmh / 255.0
        frame = frame * alpha3
        frame = frame.astype(np.uint8)
    orig = frame[:,offset:-offset,0:3].copy()
    transparentOverlay(frame, model_img, (offset, 0))
    cv2.imshow(winname="Face", mat=frame)
    k = cv2.waitKey(delay=1)
    if k == 27:
        break
    if k == 32:
        out = np.zeros((480, 480*2, 3), dtype=np.uint8)
        out[:, 0:480, :] = orig
        out[:,480:960,:] = model_img[:,:,0:3]
        cv2.imwrite('{}/{}.png'.format(args.output, models[model_index]), out)
        model_index = model_index + 1
        if model_index >= len(models):
            break
        model_img = get_model()
