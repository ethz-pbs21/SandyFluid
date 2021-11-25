import numpy as np
import taichi as ti
ti.init(arch=ti.cuda)

window_resolution = (800, 600)
title = str(window_resolution)

window = ti.ui.Window(title, window_resolution, vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()


pos = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
# centers = np.array([[0, 0, 0], [0.5, 0, 0]])
# pos.from_numpy(centers)
pos[0] = ti.Vector([0,1,0])
# pos[0] = ti.Vector([0.5, -0.5, -0.0])

camera.position(0.0, -2.0, 0.0)
camera.lookat(0.0, 0.0, 0.0)
camera.up(0.0, 0.0, 1.0)
# camera.projection_mode(ti.ui.ProjectionMode)

# camera.position(0.5, -0.5, 2)
# camera.lookat(0.5, -0.5, 0)
scene.set_camera(camera)



while window.running:
    scene.point_light(pos=(0, 0, 0), color=(1, 1, 1))
    scene.ambient_light((1, 1, 1))
    scene.particles(pos, radius=0.1)
    canvas.scene(scene)

    window.show()
