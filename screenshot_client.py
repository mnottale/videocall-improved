#! /usr/bin/env python3
import socket
import sys
import numpy as np

def connect(sockaddr='/tmp/screenshot-server.sock'):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sockaddr)
    return sock

def screenshot(sock, wid, x=-1, y=-1, w=-1, h=-1):
    query = '{} {} {} {} {}\n'.format(wid, x, y, w, h)
    sock.sendall(query.encode())
    lb = sock.recv(4)
    l = int.from_bytes(lb, 'big')
    remain = l
    payload = b''
    while remain > 0:
        more = sock.recv(remain)
        payload = payload + more
        remain = remain - len(more)
    res = np.frombuffer(payload, dtype=np.uint8)
    res = np.reshape(res, (h, w, 3))
    return res

if __name__ == '__main__':
    import cv2
    s = connect()
    while True:
        a = screenshot(s, *list(map(int, sys.argv[1:])))
        cv2.imshow(mat=a, winname='Screenshot')
        if cv2.waitKey(delay=1) == 27:
            break