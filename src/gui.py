import taichi as ti
from simulator import Simulator

@ti.data_oriented
class SimulationGUI(object):
    def __init__(
        self,
        sim: Simulator,
        title: str = "Simulator",
        window_resolution=(640, 960)) -> None:
        super().__init__()

        self.sim = sim
        self.window_resolution = window_resolution
        
        self.window = ti.ui.Window(title, window_resolution)
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
        self.sim.
        self.camera.position()

    def render(self):

        img = ti.imresize(field, *self.window_resolution)
        self.gui.set_image(img)

        self.gui.rect(topleft=[0, 1], bottomright=[1, 0])

        self.gui.circles()
        self.gui.line



    def run(self):
        while self.gui.running:

            # handle user input
            for e in self.gui.get_events(ti.GUI.PRESS):
                self.gui_event_callback(e)

            # update sim parameters
            self.sim.dt = self.time_step_slider.value
            self.sim.wind_strength = self.wind_slider.value
            self.sim.vorticity_strength = self.vorticity_slider.value
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
            self.gui.show()

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


