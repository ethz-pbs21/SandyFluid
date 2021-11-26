import taichi as ti
from simulator import Simulator
import numpy as np

@ti.data_oriented
class SimulationGUI(object):
    def __init__(self, sim: Simulator, title: str = "Simulator", resolution=(640, 960)) -> None:
        super().__init__()

        self.sim = sim

        self.bound_min = np.arary([0.0,0.0,0.0], dtype=np.float32)
        self.bound_max = self.sim.grid_extent
        self.bound_extent = self.bound_max - self.bound_min
        self.bound_center = 0.5 * self.bound_extent + self.bound_min
        # Particle visualization radius
        self.p_radius = self.sim.cell_extent * 0.5

        self.resolution = resolution
        self.window = ti.ui.Window(title, resolution)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()
        self.reset_camera()


        self.time_step_slider = self.gui.slider("Time step (t)", 1e-6, 3)
        self.time_step_slider.value = self.sim.dt

        self.max_iterations_slider = self.gui.slider("Max Iter (i)", 1, 1e6)
        self.max_iterations_slider.value = self.sim.gauss_seidel_max_iterations

        self.min_accuracy_slider = self.gui.slider("Min accuracy (a)", 1e-6, 1)
        self.min_accuracy_slider.value = self.sim.gauss_seidel_min_accuracy

        self.scalar_field_to_render = "density"
        self.field = ti.field(ti.f32, shape=self.sim.grid_size)

        # print controlls
        print("Keyboard controls")
        print("Space:\tStart/Pause simulation")
        print("r:\tReset simulation")
        print("m:\tTurn MacCormack advection ON or OFF")
        print("t/T:\tIncrement/decrement time step")
        print("v/V:\tIncrement/decrement vorticity")
        print("w/W:\tIncrement/decrement wind strength")
        print("i/I:\tIncrement/decrement Gauss-Seidel max iterations")
        print("a/A:\tIncrement/decrement Gauss-Seidel min accuracy")
        print("")


    def reset_camera(self):
        # Offset the camera position from the center of the bound along -Y
        self.camera.position(self.bound_center[0], self.bound_center[1] - self.bound_extent[1], self.bound_center[2])
        self.camera.lookat(self.bound_center[0], self.bound_center[1], self.bound_center[2])
        self.camera.up(0.0, 0.0, 1.0)
        # self.camera.fov() # todo: compute this
        # self.camera.projection_mode()
        self.scene.set_camera(self.camera)

    def render(self):
        # todo: support resize Window

        # This seems to be a bug... If no point light exists then interal error would occur
        self.scene.point_light(pos=(0, 0, 0), color=(0, 0, 0))
        # Use the ambient light to see the particles
        self.scene.ambient_light((1, 1, 1))
        # Draw the particles
        self.scene.particles(self.sim.particles_position, radius=self.p_radius)

        # todo: draw the line of the edges of the bound
        # self.scene.

        self.canvas.scene(self.scene)
        self.window.show()

    def run(self):
        while self.window.running:

            # handle user input
            for e in self.gui.get_events(ti.GUI.PRESS):
                self.gui_event_callback(e)

            # update sim parameters
            self.sim.dt = self.time_step_slider.value
            self.sim.gauss_seidel_max_iterations = int(self.max_iterations_slider.value)
            self.max_iterations_slider.value = self.sim.gauss_seidel_max_iterations
            self.sim.gauss_seidel_min_accuracy = self.min_accuracy_slider.value

            # print state
            state = {
                "Step": self.sim.cur_step,
                "t": self.sim.t,
                "dt": self.sim.dt,
                "wind": self.sim.wind_strength,
                "vort": self.sim.vorticity_strength,
                "iter": self.sim.gauss_seidel_max_iterations,
                "prec": self.sim.gauss_seidel_min_accuracy,
                "MacCormack": "ON" if self.sim.mac_cormack else "OFF",
            }
            # print(
            #     ", ".join([f"{key}: {value}" for key, value in state.items()]),
            #     end="   \r",
            # )

            # run simulation step
            self.sim.step()

            # update gui
            self.render()

    def gui_event_callback(self, event):
        if event.key == " ":
            self.sim.paused = not self.sim.paused
        elif event.key == "r":
            self.sim.reset()
        elif event.key == "m":
            self.sim.mac_cormack = not self.sim.mac_cormack
        elif event.key == "t":
            if self.gui.is_pressed("Shift"):
                self.time_step_slider.value = max(self.time_step_slider.value - 0.005, 0.005)
            else:
                self.time_step_slider.value = min(self.time_step_slider.value + 0.005, 3.0)
        elif event.key == "v":
            if self.gui.is_pressed("Shift"):
                self.vorticity_slider.value = max(self.vorticity_slider.value - 0.05, 0.0)
            else:
                self.vorticity_slider.value = min(self.vorticity_slider.value + 0.05, 1.0)
        elif event.key == "w":
            if self.gui.is_pressed("Shift"):
                self.wind_slider.value = max(self.wind_slider.value - 0.5, 0)
            else:
                self.wind_slider.value = min(self.wind_slider.value + 0.5, 10)
        elif event.key == "i":
            if self.gui.is_pressed("Shift"):
                self.max_iterations_slider.value = max(self.max_iterations_slider.value - 1000, 1)
            else:
                self.max_iterations_slider.value = min(self.max_iterations_slider.value + 1000, 1e6)
        elif event.key == "a":
            if self.gui.is_pressed("Shift"):
                self.min_accuracy_slider.value = max(self.min_accuracy_slider.value - 0.05, 1e-6)
            else:
                self.min_accuracy_slider.value = min(self.min_accuracy_slider.value + 0.05, 1.0)


