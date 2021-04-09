""" Blender script to record a bunch of diverse poses by animating the rig
    from a set of parameters
"""

bone_spec = [
    (['Spine06'], 'X', -0.4, 0.3),
    (['Spine06'], 'Y', -0.4, 0.4),
    (['Spine06'], 'Z', -0.6, 0.6),
    (['Jaw_L.002', 'Jaw_R.002'], 'X', -1.0, 0.0)
]

def make_poses(spec, min_dist, max_dist, count, output_dir):
    import random
    import os
    rnd = random.Random()
    ob = bpy.data.objects['Armature']
    for i in range(count):
        for s in spec:
            bones = s[0]
            axis = s[1]
            rmin = s[2]
            rmax = s[3]
            r = rnd.random() * (rmax-rmin) + rmin
            for b in bones:
                ob.pose.bones[b].rotation_mode = 'XYZ'
                if axis == 'X':
                    ob.pose.bones[b].rotation_euler.x = r
                elif axis == 'Y':
                    ob.pose.bones[b].rotation_euler.y = r
                else:
                    ob.pose.bones[b].rotation_euler.z = r
        v = rnd.random() * (max_dist - min_dist) + min_dist
        bpy.context.screen.areas[1].spaces[0].region_3d.view_distance = v
        bpy.context.scene.render.filepath = os.path.join(output_dir, ('render-step-{}.png'.format(i)))
        bpy.ops.render.opengl(write_still = True)

make_poses(bone_spec, 6.5, 8, 10, '/tmp/poses')