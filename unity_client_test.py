#! /usr/bin/env python3

import cv2
import numpy as np
import socket


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 12223))

angle = 0
direction = 1

while True:
    angle = angle + 2 * direction
    if angle > 45:
        direction = -1
    if angle < -45:
        direction = 1
    s.send('0 0 0 0 {} {} 0\n'.format(angle, angle).encode())
    b = s.recv(1)
    if b[0] == 1:
        buf = ''.encode()
        remain = 480 * 480 * 3
        while remain > 0:
            tb = s.recv(remain)
            buf = buf + tb
            remain -= len(tb)
        frame = np.frombuffer(buf, dtype=np.uint8)
        frame = frame.reshape((480, 480, 3))
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow(mat=frame, winname='Unity')
        if cv2.waitKey(delay=10) == 27:
            break