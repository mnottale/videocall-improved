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
import random

import bubbles.particle_effect
from bubbles.renderers.opencv_effect_renderer import OpenCVEffectRenderer

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input device number')
parser.add_argument('-o', '--output', help='output device number')
parser.add_argument('-n','--no-output', action='store_true', help='disable output')

args = parser.parse_args()


rnd = random.Random()

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

bg_imgs = list()
bg_movs = list()
for f in os.listdir('.'):
    if len(f) > 3 and f[0:3] == 'bg-':
        bg_imgs.append(f)
    if len(f) > 4 and f[0:4] == 'vbg-':
        bg_movs.append(f)

background = None
bg_imgs_pos = 0
background_mov = None
bg_movs_pos = 0

width = 640
height = 480

def next_bg_img():
    global bg_imgs
    global bg_imgs_pos
    global background
    global width
    global height
    bg_imgs_pos = (bg_imgs_pos+1)%len(bg_imgs)
    background = cv2.imread(bg_imgs[bg_imgs_pos], cv2.IMREAD_COLOR)
    background = cv2.resize(background, (width, height))
def next_bg_mov():
    global background_mov
    global bg_movs_pos
    global bg_movs
    bg_movs_pos = (bg_movs_pos+1)%len(bg_movs)
    if background_mov is not None:
        background_mov.release()
    background_mov = cv2.VideoCapture(bg_movs[bg_movs_pos])
# read the image
cap = cv2.VideoCapture(int(args.input))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
out = None
if not args.no_output:
    out = cv2.VideoWriter(
        #'appsrc ! videoconvert ! v4l2sink device=/dev/video4',
        #'appsrc ! videoconvert ! video/x-raw,format=BGRx ! identity drop-allocation=true ! v4l2sink device=/dev/video4 sync=false',
        'appsrc ! videoconvert ! video/x-raw,format=UYVY ! identity drop-allocation=true ! v4l2sink device=/dev/video{} sync=false'.format(args.output),
        #BRONK 'appsrc  ! videoconvert ! jpegenc ! image/jpeg ! v4l2sink device=/dev/video4 sync=false',
        0, 20, (width, height))

display_marks = False
display_eyecolor = False
display_eyebrowcolor = False
display_tatoo = False
display_hearts = False
display_tornado = False
replace_background = False

overlay = None
overlay_sequence = None
overlay_appear_time = None
overlay_state = None

sequence_turn = [
    {
        'duration': 1.0,
        'effects': {
                'scale': {'start': 0.1, 'stop': 1.0},
                'rotation': {'start': 0.0, 'stop': 1440.0},
        }
    },
    {
        'duration': 5.0,
        'effects': {'opacity': {'start': 1.0, 'stop': 0.0}},
    }
]

with open('bubbles/examples/hearts.json', 'r') as f:
    e_hearts = f.read()
effect_hearts = bubbles.particle_effect.ParticleEffect.load_from_dict(json.loads(e_hearts))
effect_hearts_renderer = OpenCVEffectRenderer()
effect_hearts_renderer.register_effect(effect_hearts)

with open('bubbles/examples/tornado.json', 'r') as f:
    e_tornado = f.read()
effect_tornado = bubbles.particle_effect.ParticleEffect.load_from_dict(json.loads(e_tornado))
effect_tornado_renderer = OpenCVEffectRenderer()
effect_tornado_renderer.register_effect(effect_tornado)

def init_state():
    return {
        'x': 0.5,
        'y': 0.5,
        'scale': 1.0,
        'rotation': 0.0,
        'opacity': 1.0,
    }

def next_state(sequence, state, elapsed):
    t = 0
    for step in sequence:
        if t+step['duration'] < elapsed:
            t += step['duration']
            for k,v in step['effects'].items():
                state[k] = v['stop']
        else:
            pos = elapsed - t
            ratio = pos / step['duration']
            for k,v in step['effects'].items():
                state[k] = v['start'] * (1.0-ratio) + v['stop'] * ratio
            return True
    return False

def apply_state(state, image):
    if state['scale'] != 1.0 or state['rotation'] != 0.0:
        image = rotate_image(image, state['rotation'] * 3.1415 / 180.0, state['scale'])
    if state['opacity'] != 1.0:
        image[:,:,3] = image[:,:,3] * state['opacity']
    return image

def start_overlay(idx):
    global overlay
    global overlay_sequence
    global overlay_appear_time
    global overlay_state
    choices = list()
    pfx = '{}-'.format(idx)
    for f in os.listdir('.'):
        if len(f) >= len(pfx) and f[0:len(pfx)] == pfx:
            choices.append(f)
    pickIdx = rnd.randint(0, len(choices)-1)
    overlay = cv2.imread(choices[pickIdx], cv2.IMREAD_UNCHANGED)
    overlay_sequence = sequence_turn
    overlay_appear_time = datetime.datetime.now()
    overlay_state = init_state()

def next_overlay():
    global overlay_sequence
    global overlay_state
    global overlay_appear_time
    global overlay
    cont = next_state(overlay_sequence, overlay_state, (datetime.datetime.now()-overlay_appear_time).total_seconds())
    if not cont:
        overlay = None
        return None, None, None
    return apply_state(overlay_state, overlay), overlay_state['x'], overlay_state['y']  

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
    effect_tornado.update((now-last_effect_update).total_seconds())
    effect_hearts.update((now-last_effect_update).total_seconds())
    last_effect_update = now
    if display_hearts:
        effect_target = np.zeros((200, 200, 4), dtype=np.uint8) # Image.new("RGBA", (200, 200), (0,0,0,0))
        effect_hearts_renderer.render_effect(effect_hearts, effect_target)
        el = landmarks.part(37)
        er = landmarks.part(44)
        transparentOverlay(frame, effect_target, (el.x-100, el.y-200))
        transparentOverlay(frame, effect_target, (er.x-100, er.y-200))
    if display_tornado:
        effect_target = np.zeros((200, 200, 4), dtype=np.uint8) # Image.new("RGBA", (200, 200), (0,0,0,0))
        effect_tornado_renderer.render_effect(effect_tornado, effect_target)
        el = landmarks.part(32)
        er = landmarks.part(34)
        transparentOverlay(frame, effect_target, (el.x-100, el.y-200))
        transparentOverlay(frame, effect_target, (er.x-100, er.y-200))

frame_count = 0
last_ts = datetime.datetime.now()
last_count = 0
prev_image = None
msk_hist = None
while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = detector(gray)
    landmarks = list()
    for face in faces:
        landmarks.append(predictor(image=gray, box=face))
    # Background handling
    if replace_background:
        if background_mov is not None:
            _, bg = background_mov.read()
            bg = cv2.resize(bg, (width, height))
        else:
            bg = background
        # We mask as foreground the convex hull of a cleaned version of changing
        # pixels, plus detected faces
        diff = cv2.absdiff(gray, prev_image)
        th = 12
        bmsk = diff > th
        msk = np.zeros_like(gray)
        msk[:] = bmsk * 255
        kernel = np.ones((5,5),np.uint8)
        msk = cv2.erode(msk, kernel, iterations=1)

        msk = cv2.dilate(msk, kernel, iterations=10)
        msk = cv2.erode(msk, kernel, iterations=8)
        contours, hierarchy = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            hull = cv2.convexHull(c, False)
            cv2.drawContours(msk, [hull], 0, 255, cv2.FILLED)
        for l in landmarks:
            idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            pts = list()
            for i in idxs:
                p = l.part(i)
                pts.append([p.x, p.y])
            # add symetrical points from top-nose to get the forehead
            idxs.reverse()
            mid = l.part(27)
            for i in idxs:
                p = l.part(i)
                nx = mid.x + mid.x - p.x
                ny = mid.y + mid.y - p.y
                pts.append([nx, ny])
            ctr = np.array(pts).reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(msk, [ctr], 0, 255, cv2.FILLED)
        if msk_hist is None:
            msk_hist = msk
        else:
            msk_hist = cv2.max(msk, msk_hist)
            msk_hist = cv2.subtract(msk_hist, 4)
        bmh = cv2.blur(msk_hist, (15, 15))
        #cv2.imshow(winname='Mask', mat=msk_hist)
        alpha3 = np.zeros(frame.shape, dtype=np.float64)
        alpha3[:,:,0] = bmh / 255.0
        alpha3[:,:,1] = bmh / 255.0
        alpha3[:,:,2] = bmh / 255.0
        frame = frame * alpha3 + bg * (1.0 - alpha3)
        frame = frame.astype(np.uint8)
    prev_image = gray


    for l in landmarks:
        augment(frame, l)

    if overlay is not None:
        img, x, y = next_overlay()
        if img is not None:
            transparentOverlay(frame, img, (int(frame.shape[1] * x-img.shape[1]/2), int(frame.shape[0]*y-img.shape[0]/2)))
    # show the image
    cv2.imshow(winname="Face", mat=frame)
    if out is not None:
        out.write(frame)
    # Exit when escape is pressed
    k = cv2.waitKey(delay=1)
    if k == 27:
        break
    elif k == 103: # g
        replace_background = not replace_background
        if replace_background:
            next_bg_img()
        else:
            background = None
            if background_mov is not None:
                background_mov.release()
            background_mov = None
    elif k == 118: # v
        replace_background = not replace_background
        if replace_background:
            next_bg_mov()
        else:
            background = None
            if background_mov is not None:
                background_mov.release()
            background_mov = None
    elif k == 100: # d
        display_marks = not display_marks
    elif k == 101: # e
        display_eyecolor = not display_eyecolor
    elif k == 104: # h
        display_hearts = not display_hearts
        if display_hearts:
            list(map(lambda x: x.clear(), effect_hearts.get_emitters()))
    elif k == 98: # b
        display_eyebrowcolor = not display_eyebrowcolor
    elif k == 116: # t
        display_tatoo = not display_tatoo
    elif k == 110: # n
        display_tornado = not display_tornado
        if display_tornado:
            list(map(lambda x: x.clear(), effect_tornado.get_emitters()))
    elif k == 9: # tab
        next_tatoo()
    elif k in range(48, 58): # 0-9
        idx = k - 48
        start_overlay(idx)
    elif k == 86: # pgdown
        if background is not None:
            next_bg_img()
        elif background_mov is not None:
            next_bg_mov()
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