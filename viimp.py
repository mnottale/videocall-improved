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
import socket


save = sys.path
sys.path.append("bubbles")
import bubbles.particle_effect
from bubbles.renderers.opencv_effect_renderer import OpenCVEffectRenderer
sys.path = save

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input device number')
parser.add_argument('-o', '--output', help='output device number')
parser.add_argument('-n','--no-output', action='store_true', help='disable output', default=False)
parser.add_argument('-m', '--model', help='co-mod-gan model')
parser.add_argument('-a', '--anime', help='U-GAT-IT selfie2anime model', action='store_true', default=False)
parser.add_argument('--anime-size', help='Resolution to use for anime', default='128')
parser.add_argument('-f', '--funit', help='funit NN', default=False, action='store_true')
parser.add_argument('-c', '--funit-class', help='funit class to use', default='labrador')
parser.add_argument('-r', '--record', help='record pre+post images in given dir', default=None)
parser.add_argument('-s', '--server', help='connect to given face server (host:port)', default=None)

args = parser.parse_args()

anime_size = int(args.anime_size)
latent = None
if args.model is not None:
    import dnnlib
    import dnnlib.tflib
    dnnlib.tflib.init_tf()
    from training import misc, dataset
    _, _, network = misc.load_pkl(args.model)
    latent = np.random.randn(1, *network.input_shape[1:])
    latent.fill(0)
if args.funit:
    class Foo:
        pass
    import torch
    import torch.backends.cudnn as cudnn
    from torchvision import transforms
    save = sys.path
    sys.path.append("FUNIT")
    from FUNIT.utils import get_config
    from FUNIT.trainer import Trainer
    sys.path = save
    from PIL import Image
    argsd = {
        'config': 'FUNIT/configs/funit_animals.yaml',
        'ckpt': 'FUNIT/pretrained/animal149_gen.pt',
        'class_image_folder': 'FUNIT/images/'+args.funit_class,
    }
    d = Foo()
    for (k, v) in argsd.items():
        d.__dict__[k] = v
    config = get_config(d.config)
    config['batch_size'] = 1
    config['gpus'] = 1
    trainer = Trainer(config)
    trainer.cuda()
    trainer.load_ckpt(d.ckpt)
    trainer.eval()
    transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize((128, 128))] + transform_list
    transform = transforms.Compose(transform_list)
    images = os.listdir(d.class_image_folder)
    for i, f in enumerate(images):
        fn = os.path.join(d.class_image_folder, f)
        img = Image.open(fn).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            class_code = trainer.model.compute_k_style(img_tensor, 1)
            if i == 0:
                new_class_code = class_code
            else:
                new_class_code += class_code
    final_class_code = new_class_code / len(images)
    
if args.anime:
    class Foo:
        pass
    save = sys.path
    sys.path.append("UGATIT")
    from UGATIT import UGATIT
    sys.path = save
    import tensorflow as tf
    tf_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    argsd = {
        'checkpoint_dir': 'UGATIT/checkpoint',
        'dataset': 'b2f', # 'b2l-2', # 's2a128', # 'furry', # 'selfie2anime',
        'iteration': 100,
        'print_freq': 1,
        'save_freq': 1,
        'result_dir': 'results',
        'log_dir': 'logs',
        'sample_dir': 'samples',
        'decay_epoch': 50, 
        'lr': 0.0001,
        'GP_ld': 10,
        'epoch': 1,
        'batch_size': 1,
        'phase': 'test',
        'light': False,
        'decay_flag': True,
        'adv_weight': 1,
        'cycle_weight': 10,
        'identity_weight': 10,
        'cam_weight': 1000,
        'gan_type': 'lsgan',
        'smoothing': True,
        'ch': 64,
        'n_res': 4,
        'n_dis': 6,
        'n_critic': 1,
        'sn': True,
        'img_size': anime_size, # 128, # 256,
        'img_ch': 3,
        'augment_flag': True,
    }
    d = Foo()
    for (k, v) in argsd.items():
        d.__dict__[k] = v
    gan = UGATIT(tf_session, d)
    gan.build_model()
    tf.global_variables_initializer().run(session=tf_session)
    gan.saver = tf.train.Saver()
    could_load, checkpoint_counter = gan.load(gan.checkpoint_dir)
    print('CHECKPOINT: '.format(could_load))

i2p = None
if args.server is not None:
    from PIL import Image
    save = sys.path
    sys.path.append("img2pose")
    from run_face_alignment import img2pose
    sys.path = save
    class Foo:
        pass
    argsd = {
        'max_faces': 1,
        'order_method': 'position',
        'face_size': 224,
        'min_size': 256,
        'max_size': 1024,
        'depth': 18,
        'pose_mean': 'img2pose/models/WIDER_train_pose_mean_v1.npy',
        'pose_stddev': 'img2pose/models/WIDER_train_pose_stddev_v1.npy',
        'pretrained_path': 'img2pose/models/img2pose_v1.pth',
        'threed_5_points': 'img2pose/pose_references/reference_3d_5_points_trans.npy',
        'threed_68_points': 'img2pose/pose_references/reference_3d_68_points_trans.npy',
        'nms_threshold': 0.6,
        'det_threshold': 0.7,
        'images_path': '/tmp',
        'output_path': '/dev/null',
    }
    d = Foo()
    for (k, v) in argsd.items():
        d.__dict__[k] = v
    i2p = img2pose(d)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    hostport = args.server.split(':')
    client.connect((hostport[0], int(hostport[1])))

def run_anime(img):
    global gan
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array([img])
    img = img/127.5 - 1
    for i in range(1):
        res = gan.sess.run(gan.test_fake_B, feed_dict = {gan.test_domain_A : img})
        img = res
    res = res[0,:,:,:]
    res = ((res+1.0)/2.0) * 255.0
    res = cv2.cvtColor(res.astype('uint8'), cv2.COLOR_RGB2BGR)
    return res

def run_network(image, mask=None):
    global latent
    image = np.transpose(image, (2,0,1))
    real = misc.adjust_dynamic_range(image, [0, 255], [-1, 1])
    real = np.array([real])
    if mask is None:
        mask = np.ones((1, 1, 512, 512), np.uint8)
        #mask[0, 0, 270:500, 128:384] = 0 # mouth
        mask[0, 0, 128:384, 0:256] = 0 # eye
    else:
        mask = np.array([[mask]])
    fake = network.run(latent, [1], real, mask, truncation_psi=None)
    return misc.adjust_dynamic_range(fake, [-1, 1], [0, 255]).clip(0, 255).astype(np.uint8)

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
#width = 1024
#height = 768

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
    print('Opening {}'.format(bg_movs[bg_movs_pos]))
    background_mov = cv2.VideoCapture(bg_movs[bg_movs_pos])
# read the image
cap = cv2.VideoCapture(int(args.input))
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
out = None
if not args.no_output:
    print('Opening output....' + str(args.output))
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
display_bigeye = False
display_lightning = False
display_network = False
replace_background = False
display_anime = False
display_funit = False
display_server = False

bigeye_start = 0

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

def rod(segments, displacement):
    res = [0.0, 0.0]
    for i in range(segments-1):
        next = list()
        for j in range(len(res)-1):
            avg = (res[j]+res[j+1])/2.0
            disp = (rnd.uniform(0, 1)-0.5)*displacement
            next.append(res[j])
            next.append(avg+disp)
        next.append(res[-1])
        res = next
        displacement = displacement * 0.6
    return res
def lightning(image, center, dist, count=20, segments=5, displacement=10):
    for l in range(count):
        a = 3.14159*2.0 * l / count
        r = rod(segments, displacement)
        for i in range(len(r)-1):
            sx0 = dist * i / len(r)
            sx1 = dist * (i+1) / len(r)
            sy0 = r[i]
            sy1 = r[i+1]
            x0 = center[0] + sx0 * math.cos(a) - sy0 * math.sin(a)
            x1 = center[0] + sx1 * math.cos(a) - sy1 * math.sin(a)
            y0 = center[1] + sx0 * math.sin(a) + sy0 * math.cos(a)
            y1 = center[1] + sx1 * math.sin(a) + sy1 * math.cos(a)
            cv2.line(image,
                (int(x0), int(y0)), (int(x1), int(y1)),
                (rnd.randint(235,255), rnd.randint(0, 15), rnd.randint(40, 60), 255),
                1)
def remap(image, point, distance):
    point = (int(point[0]), int(point[1]))
    hd = int(distance)
    x1 = max(0, point[0]-hd)
    x2 = min(image.shape[1]-1, point[0]+hd)
    y1 = max(0, point[1]-hd)
    y2 = min(image.shape[0]-1, point[1]+hd)
    zone = image[y1:y2,x1:x2]
    point = (point[0]-x1, point[1]-y1)
    mx = np.zeros((zone.shape[0], zone.shape[1]), dtype=np.float32)
    my = np.zeros((zone.shape[0], zone.shape[1]), dtype=np.float32)
    for y in range(zone.shape[0]):
        for x in range(zone.shape[1]):
            d = math.sqrt((point[0]-x)*(point[0]-x)+(point[1]-y)*(point[1]-y))
            if d > distance:
                mx[y,x] = x
                my[y,x] = y
            else:
                factor =  1.0 - math.log(d/distance + 1)/math.log(2)
                mx[y,x] = x + (point[0]-x)*factor
                my[y,x] = y + (point[1]-y)*factor
    zone = cv2.remap(zone, mx, my, cv2.INTER_LINEAR)
    image[y1:y2,x1:x2] = zone
    return image

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


selectedZones = list()
selectStart = None
selectRadius = 0.0
def face_click(event, x, y, flags, param):
    global selectStart, selectRadius, selectedZones
    if event == cv2.EVENT_LBUTTONDOWN:
        selectStart = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selectStart is not None:
        selectRadius = math.sqrt((x-selectStart[0])*(x-selectStart[0])+(y-selectStart[1])*(y-selectStart[1]))
    elif event == cv2.EVENT_LBUTTONUP and selectStart is not None:
        selectedZones.append((selectStart, selectRadius))
        selectStart = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i in range(len(selectedZones)):
            c = selectedZones[i][0]
            r = selectedZones[i][1]
            if (c[0]-x)*(c[0]-x)+(c[1]-y)*(c[1]-y) < r * r:
                del selectedZones[i]
                break

last_effect_update = datetime.datetime.now()
def augment(frame, landmarks):
    global last_effect_update
    global frame_count
    global bigeye_start
    if display_funit:
        hdw = (640-480) // 2
        image = frame[:,hdw:640-hdw,:]
        content_img = transform(Image.fromarray(image)).unsqueeze(0)
        with torch.no_grad():
            output_image = trainer.model.translate_simple(content_img, final_class_code)
            image = output_image.detach().cpu().squeeze().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = ((image + 1) * 0.5 * 255.0)
            image = image.astype(np.uint8)
            image = cv2.resize(image, (480, 480))
            #print(image.shape)
            #print(frame[:,hdw:640-hdw,:].shape)
            frame[:,hdw:640-hdw,:] = image
    if False and display_anime:
        #c = landmarks.part(30)
        #c2 = landmarks.part(28)
        #left = landmarks.part(2)
        #right = landmarks.part(14)
        #bottom = landmarks.part(8)
        #top = landmarks.part(19)
        #ytop = (c.y-top.y) * 2.4
        #ybottom = (bottom.y - c.y) * 1.5
        #xleft = (c.x - left.x) * 1.2
        #xright = (right.x-c.x) * 1.2
        #dx = xleft + xright
        #dy = ytop + ybottom
        #hdy = int(dy / 2)
        #print('t {} b {} h {}'.format(ytop, ybottom, hdy))
        #zone = frame[c2.y-hdy:c2.y+hdy, c2.x-hdy:c2.x+hdy, :]
        mrg = (width-height) // 2
        zone = frame[:,mrg:width-mrg,:]
        sh = zone.shape
        zone = cv2.resize(zone, (anime_size, anime_size))
        cv2.imshow(winname="Zone", mat=zone)
        if True: # sh == (hdy*2, hdy*2, 3):
            res = run_anime(zone)
            res = cv2.resize(res, (height, height))
            frame[:,mrg:width-mrg,:] = res
            #res = np.transpose(res[0,:,:,:], (1, 2, 0))
            #res = cv2.resize(res, (hdy*2, hdy*2))
            #frame[c2.y-hdy:c2.y+hdy, c2.x-hdy:c2.x+hdy, :] = res
        else:
            print('bonk {} {}'.format(hdy*2, sh))
    if display_network:
        c = landmarks.part(30)
        c2 = landmarks.part(28)
        left = landmarks.part(2)
        right = landmarks.part(14)
        bottom = landmarks.part(8)
        top = landmarks.part(19)
        ytop = (c.y-top.y) * 1.8
        ybottom = (bottom.y - c.y) * 1.2
        xleft = (c.x - left.x) * 1.2
        xright = (right.x-c.x) * 1.2
        dx = xleft + xright
        dy = ytop + ybottom
        hdy = int(dy / 2)
        print('t {} b {} h {}'.format(ytop, ybottom, hdy))
        zone = frame[c2.y-hdy:c2.y+hdy, c2.x-hdy:c2.x+hdy, :]
        sh = zone.shape
        zone = cv2.resize(zone, (512, 512))
        cv2.imshow(winname="Zone", mat=zone)
        if sh == (hdy*2, hdy*2, 3):
            msk = np.ones((sh[0], sh[1]), dtype=np.uint8)
            for z in selectedZones:
                cv2.circle(img=msk, center=(z[0][0]-c2.x+hdy, z[0][1]-c2.y+hdy), radius=int(z[1]), color=0, thickness=-1)
            msk = cv2.resize(msk, (512, 512))
            res = run_network(zone, msk)
            res = np.transpose(res[0,:,:,:], (1, 2, 0))
            res = cv2.resize(res, (hdy*2, hdy*2))
            frame[c2.y-hdy:c2.y+hdy, c2.x-hdy:c2.x+hdy, :] = res
        else:
            print('bonk {} {}'.format(hdy*2, sh))
        #cx = c.x
        #cy = c.y
        #if cx < 256:
        #    cx = 256
        #if cy < 256:
        #    cy = 256
        #if cx > frame.shape[1]-256:
        #    cx = frame.shape[1]-256
        #if cy > frame.shape[0]-256:
        #    cy > frame.shape[0]-256
        #zone = frame[cy-256:cy+256,cx-256:cx+256,:]
        #res = run_network(zone)
        #frame[cy-256:cy+256,cx-256:cx+256,:] = zone
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
    if display_lightning:
        p1 = landmarks.part(37)
        p2 = landmarks.part(40)
        lightning(frame, ((p1.x+p2.x)/2, (p1.y+p2.y)/2), 100, 40, 5, 30)
        p1 = landmarks.part(43)
        p2 = landmarks.part(46)
        lightning(frame, ((p1.x+p2.x)/2, (p1.y+p2.y)/2), 100, 40, 5, 30)
    if display_bigeye:
        p1 = landmarks.part(37)
        p2 = landmarks.part(40)
        frame = remap(frame, ((p1.x+p2.x)/2, (p1.y+p2.y)/2), 10 + min((frame_count-bigeye_start)*2,100))
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
    return frame

frame_count = 0
last_ts = datetime.datetime.now()
last_count = 0
prev_image = None
msk_hist = None

def preview_zones(image):
    for z in selectedZones:
        cv2.circle(img=image, center=z[0], radius=int(z[1]), color=(0,255,0), thickness=2)
    if selectStart is not None:
        cv2.circle(img=image, center=selectStart, radius=int(selectRadius), color=(0,0,255), thickness=2)

def r2d(r):
    return r * 180.0 / 3.14159

cv2.namedWindow("Face")
cv2.setMouseCallback("Face", face_click)
while True:
    _, frame = cap.read()
    if i2p is not None and display_server:
        mrg = (width-height) // 2
        zone = frame[:,mrg:width-mrg,:]
        start = datetime.datetime.now()
        poses = i2p.model.predict([i2p.transform(Image.fromarray(zone))])[0]
        #print('took {}'.format((datetime.datetime.now()-start).total_seconds()))
        all_scores = poses["scores"].cpu().numpy().astype("float")
        all_poses = poses["dofs"].cpu().numpy().astype("float")
        all_poses = all_poses[all_scores > i2p.det_threshold]
        all_scores = all_scores[all_scores > i2p.det_threshold]
        if len(all_poses) > 0:
            pose = all_poses[0] # rx ry rz tx ty tz in rads
            score = all_scores[0]
            #print('SCORE: {}   POSE: {}'.format(score, pose))
            client.send('0 0 0 {} {} {} 0\n'.format(*map(r2d, pose[0:3])).encode())
            b = client.recv(1)
            if b[0] == 1:
                buf = ''.encode()
                remain = 480 * 480 * 3
                while remain > 0:
                    tb = client.recv(remain)
                    buf = buf + tb
                    remain -= len(tb)
                repl = np.frombuffer(buf, dtype=np.uint8)
                repl = repl.reshape((480, 480, 3))
                repl = cv2.rotate(repl, cv2.ROTATE_180)
                repl = cv2.cvtColor(repl, cv2.COLOR_RGB2BGR)
                frame[:,mrg:width-mrg,:] = repl
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = detector(gray)
    landmarks = list()
    for face in faces:
        landmarks.append(predictor(image=gray, box=face))
    if display_anime: 
        mrg = (width-height) // 2
        zone = frame[:,mrg:width-mrg,:]
        if args.record is not None:
            orig = zone.copy()
        sh = zone.shape
        zone = cv2.resize(zone, (anime_size, anime_size))
        #cv2.imshow(winname="Zone", mat=zone)
        res = run_anime(zone)
        res = cv2.resize(res, (height, height))
        frame[:,mrg:width-mrg,:] = res
        if args.record is not None:
            rec = np.zeros((height, height*2, 3), dtype=np.uint8)
            rec[:,0:height,:] = orig
            rec[:, height:height*2,:] = res
            cv2.imwrite('{}/{:04d}.png'.format(args.record, frame_count), rec)
    # Background handling
    if replace_background:
        if background_mov is not None:
            ok, bg = background_mov.read()
            if not ok: # Assume EOF
                background_mov = cv2.VideoCapture(bg_movs[bg_movs_pos])
                ok, bg = background_mov.read()
                if not ok:
                    background_mov = None
                    bg = np.zeros_like(frame)
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
        frame = augment(frame, l)

    if overlay is not None:
        img, x, y = next_overlay()
        if img is not None:
            transparentOverlay(frame, img, (int(frame.shape[1] * x-img.shape[1]/2), int(frame.shape[0]*y-img.shape[0]/2)))
    if out is not None:
        out.write(frame)
    # show the image
    preview_zones(frame) 
    cv2.imshow(winname="Face", mat=frame)
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
    elif k == 102: # f
        display_funit = not display_funit
    elif k == 104: # h
        display_hearts = not display_hearts
        if display_hearts:
            list(map(lambda x: x.clear(), effect_hearts.get_emitters()))
    elif k == 108: # l
        display_lightning = not display_lightning
    elif k == 97: # a
        display_anime = not display_anime
    elif k == 98: # b
        display_eyebrowcolor = not display_eyebrowcolor
    elif k == 115: # s
        display_server = not display_server
    elif k == 116: # t
        display_tatoo = not display_tatoo
    elif k == 117: # u
        display_network = not display_network
    elif k == 110: # n
        display_tornado = not display_tornado
        if display_tornado:
            list(map(lambda x: x.clear(), effect_tornado.get_emitters()))
    elif k == 111: # o
        display_bigeye = not display_bigeye
        if display_bigeye:
            bigeye_start = frame_count
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
    elif k == 43: # '+'
        final_class_code = final_class_code + 1
    elif k == 45: # '-'
        final_class_code = final_class_code - 1
    elif k != 0 and k != -1:
        print('unknown key {}'.format(k))
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