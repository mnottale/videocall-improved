#! /usr/bin/env python3

import socket
import sys
import os
import json
import time

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/blender-pose-server.sock')
bone = sys.argv[1]
axis = sys.argv[2]
minv = float(sys.argv[3])
maxv = float(sys.argv[4])
step = float(sys.argv[5])
delay = float(sys.argv[6])

pos = minv
direction = 1.0
while True:
    pos = pos + step * direction
    if pos > maxv:
        direction = -1.0
    if pos < minv:
        direction = 1.0
    sock.sendall(json.dumps([[[bone], axis, pos]]).encode())
    sock.recv(1)
    time.sleep(delay)