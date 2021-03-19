#! /usr/bin/env python3

import cv2
import dlib
import numpy as np

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
# https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the image
cap = cv2.VideoCapture(2)
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
out = cv2.VideoWriter(
    #'appsrc ! videoconvert ! v4l2sink device=/dev/video4',
    #'appsrc ! videoconvert ! video/x-raw,format=BGRx ! identity drop-allocation=true ! v4l2sink device=/dev/video4 sync=false',
    'appsrc ! videoconvert ! video/x-raw,format=UYVY ! identity drop-allocation=true ! v4l2sink device=/dev/video4 sync=false',
    #BRONK 'appsrc  ! videoconvert ! jpegenc ! image/jpeg ! v4l2sink device=/dev/video4 sync=false',
    0, 20, (640, 480))

display_marks = True
display_eyecolor = True
display_eyebrowcolor = True

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

def augment(frame, landmarks):
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
    out.write(frame)
    # Exit when escape is pressed
    k = cv2.waitKey(delay=1)
    if k == 27:
        break
    elif k == 100: # d
        display_marks = not display_marks
    elif k == 101: # e
        display_eyecolor = not display_eyecolor
    elif k == 98: # b
        display_eyebrowcolor = not display_eyebrowcolor

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()