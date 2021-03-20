#! /usr/bin/env python3
import sys
import json
import cv2
import numpy as np

import bubbles.particle_effect
from bubbles.renderers.opencv_effect_renderer import OpenCVEffectRenderer


with open(sys.argv[1], 'r') as f:
    edef = f.read()
effect = bubbles.particle_effect.ParticleEffect.load_from_dict(json.loads(edef))
effect_renderer = OpenCVEffectRenderer()
effect_renderer.register_effect(effect)

while True:
    effect_target = np.zeros((500, 500, 4), dtype=np.uint8)
    effect.update(0.02)
    effect_renderer.render_effect(effect, effect_target)
    cv2.imshow(winname="Effect", mat=effect_target)
    k = cv2.waitKey(delay=20)
    if k == 27:
        break