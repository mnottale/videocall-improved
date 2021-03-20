Video conferencing improved
===========================

This projects enable you to overlay fun stuff onto your webcam video stream to
make those video calls less boring.


Requirements
------------

v4l2loopback : this is the kernel driver that allows creating fake cameras.
On ubuntu it is available as `v4l2loopback-dkms`. You need to pass the option
`exclusive_caps=1` for it to work well with web browsers:

    insmod /lib/modules/$(uname -r)/updates/dkms/v4l2loopback.ko exclusive_caps=1 video_nr=4

Face landmarks detector trained model for dlib: You can find one here: https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
Download it and drop it in the project directory.

Python3 modules: opencv, dlib, numpy

Optional: 'tatoos' images in the current directory, with any name prefixed with `tatoo-`

Optional: overlay images in the current directory, with names prefixed with `0-` to `9-`

Installing
----------

    git clone https://github.com/mnottale/videocall-improved
    git submodule update --init

Running
-------

    PYTHONPATH=bubbles: ./viimp.py -i <INPUT_CAMERA_NUMBER> -o 4

This will create a preview window on which you can type keys to enable/disable
augments.

The following augments are available, toggled by the given key:

    h: hearts coming from the eyes
    n: smoke coming from the nose
    e: blue eyes
    b: orange eyebrows (not working that well...)
    d: debug view of face tracker landmark points
    t: display image on the forehead
    <tab>: cycle through availables 'tatoo-' images
    0-9: animate in and out an image at random with prefix 0- to 9-
