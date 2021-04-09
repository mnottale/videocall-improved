#! /usr/bin/env python3

""" Listen on an UNIX socket for screenshot requests, perform them and return
    the images
"""

import numpy as np
import sys
import os
import socket
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gdk
from gi.repository import GdkPixbuf
window = Gdk.get_default_root_window()
screen = window.get_screen()

def grabit(wid, x, y, ww, hh):
    stack = screen.get_window_stack()
    hit = None
    for w in stack:
        if w.get_xid() == wid:
            hit = w
            break
    if x == -1:
        img_pixbuf = Gdk.pixbuf_get_from_window(hit, *hit.get_geometry())
    else:
        img_pixbuf = Gdk.pixbuf_get_from_window(hit, x, y, ww, hh)
    ar = pixbuf_to_array(img_pixbuf)
    return ar

def pixbuf_to_array(p):
    w,h,c,r=(p.get_width(), p.get_height(), p.get_n_channels(), p.get_rowstride())
    assert p.get_colorspace() == GdkPixbuf.Colorspace.RGB
    assert p.get_bits_per_sample() == 8
    if  p.get_has_alpha():
        assert c == 4
    else:
        assert c == 3
    assert r >= w * c
    a=np.frombuffer(p.get_pixels(),dtype=np.uint8)
    if a.shape[0] == w*c*h:
        return a.reshape( (h, w, c) )
    else:
        b=np.zeros((h,w*c),'uint8')
        for j in range(h):
            b[j,:]=a[r*j:r*j+w*c]
        return b.reshape( (h, w, c) )

def run_screenshot_server(sockaddr='/tmp/screenshot-server.sock'):
    try:
        os.unlink(sockaddr)
    except Exception:
        pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(sockaddr)
    sock.listen(1)
    while True:
        connection, client_address = sock.accept()
        try:
            while True:
                sdata = connection.recv(8192).decode('utf-8')
                sdata = sdata.split('\n')[0]
                print('QUERY: {}'.format(sdata))
                comps = sdata.split(' ')
                ar = grabit(*list(map(int, comps)))
                print('RESULT SHAPE: {}'.format(ar.shape))
                b = ar.tobytes()
                l = len(b)
                connection.sendall(l.to_bytes(4, byteorder='big'))
                connection.sendall(b)
        except Exception:
            pass
        finally:
            connection.close()

if __name__ == '__main__':
    run_screenshot_server(len(sys.argv) > 1 and sys.argv[1] or '/tmp/screenshot-server.sock')