""" Listen on an UNIX socket for bone pose and viewport commands and apply them
"""
import socket
import sys
import os
import bpy
import json

def applyspec(spec):
    ob = bpy.data.objects['Armature']
    for s in spec:
        bones = s[0]
        axis = s[1]
        angle = s[2]
        for b in bones:
            if b == 'viewport_distance':
                bpy.context.screen.areas[1].spaces[0].region_3d.view_distance = angle
            else:
                ob.pose.bones[b].rotation_mode = 'XYZ'
                if axis == 'X':
                    ob.pose.bones[b].rotation_euler.x = angle
                elif axis == 'Y':
                    ob.pose.bones[b].rotation_euler.y = angle
                else:
                    ob.pose.bones[b].rotation_euler.z = angle

def run_pose_server(sockaddr='/tmp/blender-pose-server.sock'):
    print('Running pose server')
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
                applyspec(json.loads(sdata))
                connection.sendall('1'.encode())
        except Exception as e:
            print('BRONK: {}'.format(e))
        finally:
            connection.close()
def run_pose_server_threaded(sockaddr='/tmp/blender-pose-server.sock'):
    import threading
    t = threading.Thread(target=run_pose_server, args=[sockaddr])
    t.start()
    return t