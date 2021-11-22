# Based on code provided in Exercise 4

from os import stat
from typing import Tuple
from numpy.core.fromnumeric import shape
import taichi as ti
from taichi.misc.gui import rgb_to_hex
import numpy as np
import utils

params = {
    'dt' : 0.045,   # Time step
    'mac_cormack' : False,
    'gauss_seidel_max_iterations' : 1000,
    'gauss_seidel_min_accuracy' : 1e-5,

}


@ti.func
def clear_field(f: ti.template(), v: ti.template() = 0):
    for x, y in ti.ndrange(*f.shape):
        f[x, y] = v


@ti.kernel
def abs_vector(f1: ti.template(), f2: ti.template(), fout: ti.template()):
    for x, y in ti.ndrange(*fout.shape):
        fout[x, y] = ti.sqrt(f1[x, y] ** 2 + f2[x, y] ** 2)


@ti.data_oriented
class HybridSimulator(object):
    def __init__(
        self,
        params : dict
    ):

        def get_param(key:str, default_val=None):
            return params[key] if key in params else default_val

        # Time step
        self.dt = get_param('dt')
        # Body force (gravity)
        self.g = ti.Vector([0.0, 0.0, -9.8])

        # 
        self.gauss_seidel_max_iterations = int(gauss_seidel_max_iterations)
        self.gauss_seidel_min_accuracy = gauss_seidel_min_accuracy

        self.paused = paused

        # parameters that are fixed (changing these after starting the simulatin will not have an effect!)
        self.resolution = resolution
        self.rho = 1.0 # todo: unit?
        self.dx = 1.0 / self.resolution[0]

        # simulation state
        self.cur_step = 0
        self.t = 0.0
        # self.r_wind = ti.Vector.field(1, ti.i32, shape=(1))

        # scalar fields
        self.density = ti.field(ti.f32, shape=self.resolution)
        self.density_forward = ti.field(ti.f32, shape=self.resolution)
        self.density_backward = ti.field(ti.f32, shape=self.resolution)
        self.density_old = ti.field(ti.f32, shape=self.resolution)

        self.pressure = ti.field(ti.f32, shape=self.resolution)

        self.divergence = ti.field(ti.f32, shape=self.resolution)

        # self.vorticity = ti.field(ti.f32, shape=self.resolution)

        # Particles
        self.particles_position = ti.Vector.field(3, dtype=ti.f32, shape=)
        self.particles_velocity = 

        # MAC Grids
        self.velocity_x = ti.field(
            ti.f32, shape=(self.resolution[0] + 1, self.resolution[1], self.resolution[2])
        )
        self.velocity_y = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1] + 1, self.resolution[2])
        )
        self.velocity_z = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1], self.resolution[2] + 1)
        )

        self.velocity_x_forward = ti.field(
            ti.f32, shape=(self.resolution[0] + 1, self.resolution[1])
        )
        self.velocity_y_forward = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1] + 1)
        )
        self.velocity_z_forward = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1], self.resolution[2] + 1)
        )

        self.velocity_x_backward = ti.field(
            ti.f32, shape=(self.resolution[0] + 1, self.resolution[1])
        )
        self.velocity_y_backward = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1] + 1)
        )
        self.velocity_z_backward = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1], self.resolution[2] + 1)
        )

        self.velocity_x_old = ti.field(
            ti.f32, shape=(self.resolution[0] + 1, self.resolution[1])
        )
        self.velocity_y_old = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1] + 1)
        )
        self.velocity_z_old = ti.field(
            ti.f32, shape=(self.resolution[0], self.resolution[1], self.resolution[2] + 1)
        )


        self.reset()

    def reset(self):
        self._reset()

        # reset simulation state
        self.cur_step = 0
        self.t = 0.0

    @ti.kernel
    def _reset(self):
        # zero out all fields
        clear_field(self.density)
        clear_field(self.density_forward)
        clear_field(self.density_backward)
        clear_field(self.density_old)
        clear_field(self.pressure)
        clear_field(self.divergence)
        clear_field(self.velocity_x)
        clear_field(self.velocity_y)
        clear_field(self.velocity_z)
        clear_field(self.velocity_x_forward)
        clear_field(self.velocity_y_forward)
        clear_field(self.velocity_z_forward)
        clear_field(self.velocity_x_backward)
        clear_field(self.velocity_y_backward)
        clear_field(self.velocity_z_backward)
        clear_field(self.velocity_x_old)
        clear_field(self.velocity_y_old)
        clear_field(self.velocity_z_old)

        # add initial density
        self.apply_density_source()

    def step(self, dt):
        # Apply body force
        self.apply_force(dt)

        # Particle to grid
        self.p2g()

        #apply_bc()

        #extrap_velocity()
        #apply_bc()

        solve_pressure()
        apply_pressure()

        #extrap_velocity()
        apply_bc()

        if use_flip:
            update_from_grid()
            advect_markers(dt)
            apply_markers()

            ux.fill(0.0)
            uy.fill(0.0)
            ux_temp.fill(0.0)
            uy_temp.fill(0.0)
            transfer_to_grid(ux_temp, uy_temp)  # reuse buffers

            save_velocities()

        else:
            advect_markers(dt)
            apply_markers()

            advect(ux_temp, ux, dt, 0.0, 0.5, nx + 1, ny)
            advect(uy_temp, uy, dt, 0.5, 0.0, nx, ny + 1)
            ux.copy_from(ux_temp)
            uy.copy_from(uy_temp)
            apply_bc()


    # def step(self):
    #     if self.paused:
    #         return
    #     self.advance(
    #         float(self.dt),
    #         float(self.wind_strength),
    #         float(self.vorticity_strength),
    #         1 if self.mac_cormack else 0,
    #         int(self.gauss_seidel_max_iterations),
    #         float(self.gauss_seidel_min_accuracy),
    #         1 if self.compute_final_divergence else 0,
    #         1 if self.compute_final_vorticity else 0,
    #     )
    #     self.t += self.dt
    #     self.cur_step += 1

    @ti.kernel
    def advance(
        self,
        dt: ti.f32,
        mac_cormack: ti.i32,
        gauss_seidel_max_iterations: ti.i32,
        gauss_seidel_min_accuracy: ti.f32,
        compute_final_divergence: ti.i32,
        compute_final_vorticity: ti.i32,
    ):


        # apply forces to current velocities
        self.apply_force(dt)

        # solve pressure equation
        self.solve_pressure(dt, gauss_seidel_max_iterations, gauss_seidel_min_accuracy)

        # project velocity to be divergence free
        self.project_velocity(dt)

        # advect density and velocity
        self.advect_values(dt, mac_cormack)

        # compute final divergence and vorticiy if necessary (this can be used for debugging)
        if compute_final_divergence != 0:
            self.compute_divergence()
        if compute_final_vorticity != 0:
            self.compute_vorticity()




    @ti.kernel
    def apply_force(self, dt: ti.f32):
        # only consider body force (gravity) for now
        self.velocity_x += dt * self.
        



    # velocity projection
    @ti.func
    def solve_pressure(
        self,
        dt: ti.f32,
        gauss_seidel_max_iterations: ti.i32,
        gauss_seidel_min_accuracy: ti.f32,
    ):

        # set Neumann boundary conditons
        for y in range(0, self.resolution[1]):
            self.velocity_x[0, y] = self.velocity_x[2, y]
            self.velocity_x[-1, y] = self.velocity_x[-3, y]
        for x in range(0, self.resolution[0]):
            self.velocity_y[x, 0] = self.velocity_y[x, 2]
            self.velocity_y[x, -1] = self.velocity_y[x, -3]

        # set tangential velocities along edges to zero
        for x in range(0, self.resolution[0] + 1):
            self.velocity_x[x, 0] = 0
            self.velocity_x[x, -1] = 0
        for y in range(0, self.resolution[1] + 1):
            self.velocity_y[0, y] = 0
            self.velocity_y[-1, y] = 0

        # compute divergence
        self.compute_divergence()

        # copy border pressures
        for y in range(0, self.resolution[1] + 1):
            self.pressure[0, y] = self.pressure[1, y]
            self.pressure[-1, y] = self.pressure[-2, y]
        for x in range(0, self.resolution[0] + 1):
            self.pressure[x, 0] = self.pressure[x, 1]
            self.pressure[x, -1] = self.pressure[x, -2]

        # solve possiion equation
        gauss_seidel_poisson_solver(
            self.pressure,
            self.divergence,
            dt,
            self.dx,
            gauss_seidel_max_iterations,
            gauss_seidel_min_accuracy,
            self.rho,
        )

    @ti.func
    def project_velocity(self, dt: ti.f32):
        velocity_projection(
            self.velocity_x, self.velocity_y, self.pressure, dt, self.dx
        )

    # advection

    @ti.func
    def advect_values(self, dt: ti.f32, mac_cormack: ti.i32):

        # save old values
        copy_field(self.density, self.density_old)
        copy_field(self.velocity_x, self.velocity_x_old)
        copy_field(self.velocity_y, self.velocity_y_old)

        # advect values forward
        advect_density(
            self.density,
            self.density_forward,
            self.velocity_x,
            self.velocity_y,
            dt,
            self.dx,
        )
        advect_velocity_x(
            self.velocity_x,
            self.velocity_x_forward,
            self.velocity_x,
            self.velocity_y,
            dt,
            self.dx,
        )
        advect_velocity_y(
            self.velocity_y,
            self.velocity_y_forward,
            self.velocity_x,
            self.velocity_y,
            dt,
            self.dx,
        )

        # copy forward values to current buffers
        copy_field(self.density_forward, self.density)
        copy_field(self.velocity_x_forward, self.velocity_x)
        copy_field(self.velocity_y_forward, self.velocity_y)

        # do MacCormack update if necessary
        if mac_cormack != 0:
            self.mac_cormack_update(dt)
            self.mac_cormack_clamp(dt)

    @ti.func
    def mac_cormack_update(self, dt: ti.f32):

        mac_cormack_update(
            self.density,
            self.velocity_x,
            self.velocity_y,
            self.density_backward,
            self.density_old,
            self.velocity_x_backward,
            self.velocity_x_old,
            self.velocity_y_backward,
            self.velocity_y_old,
            dt,
            self.dx,
        )

    @ti.func
    def mac_cormack_clamp(self, dt: ti.f32):
        mac_cormack_clamp_density(
            self.density,
            self.density_old,
            self.density_forward,
            self.velocity_x_old,
            self.velocity_y_old,
            dt,
            self.dx,
        )
        mac_cormack_clamp_velocity_x(
            self.velocity_x,
            self.velocity_x_old,
            self.velocity_x_forward,
            self.velocity_x_old,
            self.velocity_y_old,
            dt,
            self.dx,
        )
        mac_cormack_clamp_velocity_x(
            self.velocity_y,
            self.velocity_y_old,
            self.velocity_y_forward,
            self.velocity_x_old,
            self.velocity_y_old,
            dt,
            self.dx,
        )

    # divergence and vorticity

    @ti.func
    def compute_divergence(self):
        for x, y in ti.ndrange(
            (1, self.resolution[0] - 1), (1, self.resolution[1] - 1)
        ):
            dudx = (self.velocity_x[x + 1, y] - self.velocity_x[x, y]) / self.dx
            dudy = (self.velocity_y[x, y + 1] - self.velocity_y[x, y]) / self.dx
            self.divergence[x, y] = dudx + dudy

    @ti.func
    def compute_vorticity(self):
        for x, y in ti.ndrange(
            (2, self.resolution[0] - 2), (2, self.resolution[1] - 2)
        ):
            dudy = (
                (self.velocity_x[x, y + 1] - self.velocity_x[x, y - 1]) / self.dx * 0.5
            )
            dvdx = (
                (self.velocity_y[x + 1, y] - self.velocity_y[x - 1, y]) / self.dx * 0.5
            )
            self.vorticity[x, y] = dvdx - dudy


@ti.data_oriented
class SimulationGUI(object):
    def __init__(
        self,
        sim: HybridSimulator,
        title: str = "HybridSimulator",
        window_resolution=(640, 960)) -> None:
        super().__init__()

        self.sim = sim
        self.window_resolution = window_resolution
        self.gui = ti.GUI(title, window_resolution)

        self.time_step_slider = self.gui.slider("Time step (t)", 1e-6, 3)
        self.time_step_slider.value = self.sim.dt

        self.vorticity_slider = self.gui.slider("Vorticity (v)", 0, 1)
        self.vorticity_slider.value = self.sim.vorticity_strength

        self.wind_slider = self.gui.slider("Wind (w)", 0, 10)
        self.wind_slider.value = self.sim.wind_strength

        self.max_iterations_slider = self.gui.slider("Max Iter (i)", 1, 1e6)
        self.max_iterations_slider.value = self.sim.gauss_seidel_max_iterations

        self.min_accuracy_slider = self.gui.slider("Min accuracy (a)", 1e-6, 1)
        self.min_accuracy_slider.value = self.sim.gauss_seidel_min_accuracy

        self.scalar_field_to_render = "density"
        self.field = ti.field(ti.f32, shape=self.sim.resolution)

        # print controlls
        utils.print_pbs_logo()
        print("Keyboard controls")
        print("Space:\tStart/Pause simulation")
        print("r:\tReset simulation")
        print("m:\tTurn MacCormack advection ON or OFF")
        print("t/T:\tIncrement/decrement time step")
        print("v/V:\tIncrement/decrement vorticity")
        print("w/W:\tIncrement/decrement wind strength")
        print("i/I:\tIncrement/decrement Gauss-Seidel max iterations")
        print("a/A:\tIncrement/decrement Gauss-Seidel min accuracy")
        print("1:\tShow density")
        print("2:\tShow pressure")
        print("3:\tShow divergence")
        print("4:\tShow vorticity")
        print("5:\tShow velocity in x direction")
        print("6:\tShow velocity in y direction")
        print("7:\tShow absolute velocity")
        print("8:\tShow force in x direction")
        print("9:\tShow force in y direction")
        print("0:\tShow absolute force")
        print("")

    def render(self):
        if self.scalar_field_to_render == "density":
            field = self.sim.density.to_numpy()
        elif self.scalar_field_to_render == "pressure":
            field = self.sim.pressure.to_numpy()
        elif self.scalar_field_to_render == "divergence":
            field = self.sim.divergence.to_numpy()
        elif self.scalar_field_to_render == "vorticity":
            field = self.sim.vorticity.to_numpy()
        elif self.scalar_field_to_render == "velocity_x":
            field = self.sim.velocity_x.to_numpy()
        elif self.scalar_field_to_render == "velocity_y":
            field = self.sim.velocity_y.to_numpy()
        elif self.scalar_field_to_render == "velocity_abs":
            vx = self.sim.velocity_x.to_numpy()[:-1, :]
            vy = self.sim.velocity_y.to_numpy()[:, :-1]
            field = np.sqrt(np.power(vx, 2) + np.power(vy, 2))
        elif self.scalar_field_to_render == "force_x":
            field = self.sim.force_x.to_numpy()
        elif self.scalar_field_to_render == "force_y":
            field = self.sim.force_y.to_numpy()
        elif self.scalar_field_to_render == "force_abs":
            fx = self.sim.force_x.to_numpy()[:-1, :]
            fy = self.sim.force_y.to_numpy()[:, :-1]
            field = np.sqrt(np.power(fx, 2) + np.power(fy, 2))

        low, high = np.min(field), np.max(field)
        high = high if high != low else (low + 1)
        field = (field - low) / (high - low)

        img = ti.imresize(field, *self.window_resolution)
        self.gui.set_image(img)

        self.gui.rect(topleft=[0, 1], bottomright=[1, 0])

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
        elif event.key == "1":
            self.scalar_field_to_render = "density"
        elif event.key == "2":
            self.scalar_field_to_render = "pressure"
        elif event.key == "3":
            self.scalar_field_to_render = "divergence"
        elif event.key == "4":
            self.scalar_field_to_render = "vorticity"
        elif event.key == "5":
            self.scalar_field_to_render = "velocity_x"
        elif event.key == "6":
            self.scalar_field_to_render = "velocity_y"
        elif event.key == "7":
            self.scalar_field_to_render = "velocity_abs"
        elif event.key == "8":
            self.scalar_field_to_render = "force_x"
        elif event.key == "9":
            self.scalar_field_to_render = "force_y"
        elif event.key == "0":
            self.scalar_field_to_render = "force_abs"
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


