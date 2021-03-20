#! /usr/bin/env python3

import os
import math
import json
import cv2
import dlib
import numpy as np
import sys
import argparse
import datetime

from PIL import Image
import bubbles.particle_effect
from bubbles.renderers.opencv_effect_renderer import OpenCVEffectRenderer

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input device number')
parser.add_argument('-o', '--output', help='output device number')
parser.add_argument('-n','--no-output', action='store_true', help='disable output')

args = parser.parse_args()


# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
# https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



tatoos = list()
for f in os.listdir('.'):
    if len(f) > 5 and f[0:5] == 'tatoo':
        tatoos.append(f)
print('tatoos: {}'.format(' '.join(tatoos)))
tp = 0
tatoo = cv2.imread(tatoos[0], cv2.IMREAD_UNCHANGED)

# read the image
cap = cv2.VideoCapture(int(args.input))
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
out = None
if not args.no_output:
    out = cv2.VideoWriter(
        #'appsrc ! videoconvert ! v4l2sink device=/dev/video4',
        #'appsrc ! videoconvert ! video/x-raw,format=BGRx ! identity drop-allocation=true ! v4l2sink device=/dev/video4 sync=false',
        'appsrc ! videoconvert ! video/x-raw,format=UYVY ! identity drop-allocation=true ! v4l2sink device=/dev/video{} sync=false'.format(args.output),
        #BRONK 'appsrc  ! videoconvert ! jpegenc ! image/jpeg ! v4l2sink device=/dev/video4 sync=false',
        0, 20, (640, 480))

display_marks = False
display_eyecolor = False
display_eyebrowcolor = False
display_tatoo = False
display_hearts = True

with open('bubbles/examples/hearts.json', 'r') as f:
    e_hearts = f.read()
effect = bubbles.particle_effect.ParticleEffect.load_from_dict(json.loads(e_hearts))
effect_renderer = OpenCVEffectRenderer()
effect_renderer.register_effect(effect)

def next_tatoo():
    global tp
    global tatoo
    tp = (tp+1)%len(tatoos)
    tatoo = cv2.imread(tatoos[tp], cv2.IMREAD_UNCHANGED)

def color_eye(frame, p1, p2):
    #print('EYE: x1 {} x2 {} y1 {} y2 {}'.format(p1.x, p2.x, p1.y, p2.y))
    if p2.x <= p1.x or p2.y <= p1.y:
        return
    eye = frame[p1.y:p2.y, p1.x:p2.x]
    mask = cv2.inRange(eye, np.array([0,0,0]), np.array([127, 127, 127]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(eye, [c], -1, (255, 50, 50), cv2.FILLED)
    frame[p1.y:p2.y, p1.x:p2.x] = eye

def color_eyebrow(frame, ps):
    pxs = list(map(lambda p: p.x, ps))
    pxs.sort()
    pys = list(map(lambda p: p.y, ps))
    pys.sort()
    p1x = pxs[0]
    p2x = pxs[-1]
    p1y = pys[0]
    p2y = pys[-1]
    if p2x <= p1x or p2y <= p1y:
        return
    eb = frame[p1y:p2y,p1x:p2x]
    mask = cv2.inRange(eb, np.array([0,0,0]), np.array([130, 130, 130]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(eb, [c], -1, (50, 100, 200), cv2.FILLED)
    frame[p1y:p2y,p1x:p2x] = eb

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

def transparentOverlaySlow(src , overlay , pos=(0,0)):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i,y+j] = alpha*overlay[i,j,:3]+(1-alpha)*src[x+i,y+j]
    return src

def rotate_image(image, angle, scale):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle * 180.0 / 3.14159265, scale)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def embed_tatoo(frame, l, r):
    dx = r.x - l.x
    dy = r.y - l.y
    angle = math.atan2(dy, dx) # rad
    dist = math.sqrt(dx*dx+dy*dy)
    #print('A {}  D {}'.format(angle * 180.0 / 3.14, dist))
    midx = (r.x + l.x) / 2
    midy = (r.y + l.y) / 2
    scale = dist * 1.0 / tatoo.shape[1]
    rotated = rotate_image(tatoo, -angle, scale)
    cropy = int((tatoo.shape[0]-tatoo.shape[0]*scale)/2.0)
    cropx = int((tatoo.shape[1]-tatoo.shape[1]*scale)/2.0)
    croped = rotated[cropy:-cropy,cropx:-cropx]
    centerx = midx + math.sin(angle)*dist*1.2
    centery = midy - math.cos(angle)*dist*1.2
    embedx = centerx - croped.shape[1]/2.0
    embedy = centery - croped.shape[0]/2.0
    transparentOverlay(frame, croped, (int(embedx), int(embedy)))

last_effect_update = datetime.datetime.now()
def augment(frame, landmarks):
    global last_effect_update
    if display_marks:
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # Draw a circle
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
    if display_eyecolor:
        el1 = landmarks.part(37)
        el2 = landmarks.part(40)
        er1 = landmarks.part(43)
        er2 = landmarks.part(46)
        color_eye(frame, el1, el2)
        color_eye(frame, er1, er2)
    if display_eyebrowcolor:
        color_eyebrow(frame, [landmarks.part(17), landmarks.part(18), landmarks.part(19), landmarks.part(20), landmarks.part(21)])
        color_eyebrow(frame, [landmarks.part(22), landmarks.part(23), landmarks.part(24), landmarks.part(25), landmarks.part(26)])
    if display_tatoo:
        embed_tatoo(frame, landmarks.part(39), landmarks.part(42))
    now = datetime.datetime.now()
    effect.update((now-last_effect_update).total_seconds())
    last_effect_update = now
    if display_hearts:
        effect_target = np.zeros((200, 200, 4), dtype=np.uint8) # Image.new("RGBA", (200, 200), (0,0,0,0))
        effect_renderer.render_effect(effect, effect_target)
        lip = landmarks.part(66)
        transparentOverlay(frame, effect_target, (lip.x-100, lip.y-200))

frame_count = 0
last_ts = datetime.datetime.now()
last_count = 0
while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point

        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        augment(frame, landmarks)
    # show the image
    cv2.imshow(winname="Face", mat=frame)
    if out is not None:
        out.write(frame)
    # Exit when escape is pressed
    k = cv2.waitKey(delay=1)
    if k == 27:
        break
    elif k == 100: # d
        display_marks = not display_marks
    elif k == 101: # e
        display_eyecolor = not display_eyecolor
    elif k == 104: # h
        display_hearts = not display_hearts
    elif k == 98: # b
        display_eyebrowcolor = not display_eyebrowcolor
    elif k == 116: # t
        display_tatoo = not display_tatoo
    elif k == 9: # tab
        next_tatoo()
    frame_count += 1
    now = datetime.datetime.now()
    if (now - last_ts).total_seconds() > 5:
        print("FPS: {}".format((frame_count - last_count) / (now-last_ts).total_seconds()))
        last_ts = now
        last_count = frame_count

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()