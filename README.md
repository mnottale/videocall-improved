Video conferencing improved for Linux
=====================================

This project enables you to overlay fun stuff onto your webcam video stream to
make those video calls less boring.


Requirements
------------

v4l2loopback : this is the kernel driver that allows creating fake cameras.
On Ubuntu it is available as package `v4l2loopback-dkms`. You need to pass the option
`exclusive_caps=1` for it to work well with web browsers:

    insmod /lib/modules/$(uname -r)/updates/dkms/v4l2loopback.ko exclusive_caps=1 video_nr=4

Face landmarks detector trained model for dlib: You can find one here: https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
Download it and drop it in the project directory.

Python3 modules: opencv-python (with gstreamer enabled), dlib, numpy

Optional: 'tatoos' images in the project directory, with any name prefixed by `tatoo-`

Optional: overlay images in the project directory, with names prefixed by `0-` to `9-`

Optional: background images in the project directory, with names prefixed by `bg-`

Optional: background videos in the project directory, with names prefixed by `vbg-`

Optional: for UGATIT (selfie2anime): tensorflow 1.x (which means python <= 3.7),  with GPU support recommended (CUDA+CuDNN 10.0)

Installing
----------

    git clone https://github.com/mnottale/videocall-improved
    git submodule update --init

Running
-------

    ./viimp.py -i <INPUT_CAMERA_NUMBER> -o 4

You can use `v4l2-ctl --list-devices` to figure out the device number of your webcam.

This will create a preview window on which you can type keys to enable/disable
augments.

The following augments are available, toggled by the given key:

    h: hearts coming from the eyes
    n: smoke coming from the nose
    e: blue eyes
    b: orange eyebrows (not working that well...)
    d: debug view of face tracker landmark points
    l: lightning from the eyes
    t: display image on the forehead
    <tab>: cycle through availables 'tatoo-' images
    o: big eye O_o
    0-9: animate in and out an image at random with prefix 0- to 9-
    g: enable image background
    v: enable video background
    PGDOWN: cycle to next image or video background
    a: selfie2anime

Running with UGATIT (selfie2anime)
----------------------------------

You need python 3.7 and tensorflow 1.x.

Clone UGATIT under this repository (taki0112/UGATIT).

Download the selfie2anime checkpoint (256x256 resolution, needs a big GPU with >4G ram)
or train your own model (128x128 recommended).

Edit the big dictionary 'argsd' in the 'if args.anime' section of viimp.py
to pick your model name in the 'dataset' field.

Then run viimp:

    ./viimp.py -i 0 -o 4 -a --anime-size 128

and press 'a' to enable


Running with unity server (WIP)
-------------------------------

The gist of this mode is to have an Unity app running that listens for pose
commands over the network, and sends back pictures, replacing your whole image
with a 3D model that will follow your movements.

To use it, make in Unity a scene with the rigged model of your choice, add
the NetControl.cs behaviour on any component, and edit it to match your model and
rig.

Clone img2pose (vitoralbiero/img2pose) under this repository.

Then follow the img2pose repository instructions on how to download the checkpoints.

Then run your unity app (use 'build and run' recommended in windowed small resolution mode, if you run in editor
mode unity will use 100% of your GPU which will slow down image2pose).

Finally run viimp.py :

    ./viimp.py -i 0 -o 4 -s localhost:12223

and press 's' to enable

Misc utilities
--------------

augment.py : augment a dataset by scaling and rotating images

blender_pose_server.py : blender script to move a rig armature based on network commands

blender_pose_client_test.py : test client for the above

gen.py : grab images from your webcam and make a dummy face mask for pix2pix testing

grabalot.py : grab and save images from a webcam stream every 300ms

nancounter.py : blend a texture background to images with alpha channel

record_poses.py : blender script to record a bunch of random poses for a rig

screenshot_server.py : take commands of the form (windowId, rect) and return a screenshot over the network

screenshot_client.py : test client for the above

trainmap.py : pix2pix dataset creation helper: overlay target pose on webcam stream and grabs pictures on 'space'

unity_client_test.py : test client for the unity image server

NetControl.cs : Unity script to move rig and send images over the network