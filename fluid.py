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





@ti.kernel
def abs_vector(f1: ti.template(), f2: ti.template(), fout: ti.template()):
    for x, y in ti.ndrange(*fout.shape):
        fout[x, y] = ti.sqrt(f1[x, y] ** 2 + f2[x, y] ** 2)


@ti.data_oriented
class HybridSimulator(object):
    def __init__(self, params : dict):
        def get_param(key:str, default_val=None):
            return params[key] if key in params else default_val

        # Number of particles
        self.num_particles = # todo

        # Time step
        self.dt = get_param('dt')
        # Body force (gravity)
        self.g = ti.Vector([0.0, 0.0, -9.8])

        # 
        self.gauss_seidel_max_iterations = int(gauss_seidel_max_iterations)
        self.gauss_seidel_min_accuracy = gauss_seidel_min_accuracy

        self.paused = paused

        # parameters that are fixed (changing these after starting the simulatin will not have an effect!)
        self.grid_size = ti.Vector(get_param('grid_size'))
        self.inv_grid_size = 1.0 / ti.Vector

        self.rho = 1.0 # todo: unit?
        self.dx = 1.0 / self.grid_size[0]

        # simulation state
        self.cur_step = 0
        self.t = 0.0
        # self.r_wind = ti.Vector.field(1, ti.i32, shape=(1))

        self.reset()

    def init_fields(self):
        # Particles
        self.particles_position = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particles_velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)

        # MAC grid
        # self.pressure = ti.field(ti.f32, shape=self.resolution)
        self.grid_velocity_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_velocity_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_velocity_z = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))
        # For renormalization after p2g
        self.grid_weight_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_weight_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_weight_z = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))

        # self.divergence = ti.field(ti.f32, shape=self.grid_size)




    def reset(self):
        # Reset simulation state
        self.cur_step = 0
        self.t = 0.0


    def step(self, dt):
        self.cur_step += 1
        self.t += dt

        # Apply body force
        self.apply_force()

        # Scatter properties (mainly velocity) from particle to grid
        self.p2g()

        # Solve the poisson equation to get pressure
        self.solve_pressure()

        # Accelerate velocity using the solved pressure
        self.project_velocity()

        # Gather properties (mainly velocity) from grid to particle
        self.g2p()

        # Enforce boundary condition
        self.enforce_boundary_condition()

        # Advect particles
        self.advect_particles()

    # ###########################################################
    # Kernels launched in one step
    @ti.kernel
    def apply_force(self):
        # only consider body force (gravity) for now
        for p in self.particles_velocity:
            self.particles_velocity[p] += self.dt * self.g

    @ti.kernel
    def p2g(self):
        for p in self.particles_position:
            xp = self.particles_position[p]
            vp = self.particles_velocity[p]
            base = ti.floor(xp)
            frac = xp - base
            self.interp_grid(base, frac, vp)

        for i, j, k in self.grid_velocity_x:
            self.grid_velocity_x[]
                
    @ti.kernel
    def solve_pressure(
        self,
        dt: ti.f32,
        gauss_seidel_max_iterations: ti.i32,
        gauss_seidel_min_accuracy: ti.f32,
    ):

        # set Neumann boundary conditons
        for y in range(0, self.grid_size[1]):
            self.grid_velocity_x[0, y] = self.grid_velocity_x[2, y]
            self.grid_velocity_x[-1, y] = self.grid_velocity_x[-3, y]
        for x in range(0, self.grid_size[0]):
            self.grid_velocity_y[x, 0] = self.grid_velocity_y[x, 2]
            self.grid_velocity_y[x, -1] = self.grid_velocity_y[x, -3]

        # set tangential velocities along edges to zero
        for x in range(0, self.grid_size[0] + 1):
            self.grid_velocity_x[x, 0] = 0
            self.grid_velocity_x[x, -1] = 0
        for y in range(0, self.grid_size[1] + 1):
            self.grid_velocity_y[0, y] = 0
            self.grid_velocity_y[-1, y] = 0

        # compute divergence
        self.compute_divergence()

        # copy border pressures
        for y in range(0, self.grid_size[1] + 1):
            self.pressure[0, y] = self.pressure[1, y]
            self.pressure[-1, y] = self.pressure[-2, y]
        for x in range(0, self.grid_size[0] + 1):
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


    @ti.kernel
    def project_velocity(self, dt: ti.f32):
        velocity_projection(
            self.grid_velocity_x, self.grid_velocity_y, self.pressure, dt, self.dx
        )

    @ti.kernel
    def g2p():
        pass


    @ti.kernel
    def enforce_boundary_condition(self):
        pass

    @ti.kernel
    def advect_particles(self):
        pass

    # ###########################################################
    # Funcs called by kernels

    # Spline functions for interpolation
    # Input x should be non-negative (abs)

    # Quadratic B-spline
    # 0.75-x^2,         |x| in [0, 0.5)
    # 0.5*(1.5-|x|)^2,  |x| in [0.5, 1.5)
    # 0,                |x| in [1.5, inf)
    @ti.func
    def quadratic_kernel(x : float):
        if x < 0.5:
            return 0.75 - x**2
        elif x < 1.5:
            return 0.5 * (1.5 - x)**2
        else:
            return 0.0


    
    @ti.func
    def interp_grid(self, base : ti.Matrix, frac : ti.Matrix, vp : ti.Matrix):
        
        # Quadratic
        # todo: try other kernels (linear, cubic, ...)

        # Index on sides
        idx_side = [base-1, base, base+1, base+2]
        # Distance to sides
        dist_side = [1.0+frac, frac, 1.0-frac, 2.0-frac]
        # Index on centers
        idx_center = [base-1, base, base+1]
        # Distance to centers
        dist_center = [0.5+frac, ti.abs(0.5-frac), 1.5-frac]


        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    w = self.quadratic_kernel(dist_side[i].x) * self.quadratic_kernel(dist_center[j].y)*self.quadratic_kernel(dist_center[k].z)
                    idx = (idx_side[i].x, idx_center[j].y, idx_center[k].z)
                    self.grid_velocity_x[idx] += vp.x * w
                    self.grid_weight_x[idx] += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                for k in ti.static(range(3)):
                    w = self.quadratic_kernel(dist_center[i].x) * self.quadratic_kernel(dist_side[j].y)*self.quadratic_kernel(dist_center[k].z)
                    idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                    self.grid_velocity_y[idx] += vp.y * w
                    self.grid_weight_y[idx] += w
        
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(4)):
                    w = self.quadratic_kernel(dist_center[i].x) * self.quadratic_kernel(dist_center[j].y)*self.quadratic_kernel(dist_side[k].z)
                    idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                    self.grid_velocity_z[idx] += vp.z * w
                    self.grid_weight_z[idx] += w


    @ti.func
    def compute_divergence(self):
        for i, j, k in ti.ndrange(
            (1, self.grid_size[0] - 1), (1, self.grid_size[1] - 1), (1, self.grid_size[2] - 1)
        ):
            dudx = (self.grid_velocity_x[x + 1, y] - self.grid_velocity_x[x, y]) / self.dx
            dudy = (self.grid_velocity_y[x, y + 1] - self.grid_velocity_y[x, y]) / self.dx
            self.divergence[x, y] = dudx + dudy




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
        self.field = ti.field(ti.f32, shape=self.sim.grid_size)

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
        elif self.scalar_field_to_render == "grid_velocity_x":
            field = self.sim.grid_velocity_x.to_numpy()
        elif self.scalar_field_to_render == "grid_velocity_y":
            field = self.sim.grid_velocity_y.to_numpy()
        elif self.scalar_field_to_render == "velocity_abs":
            vx = self.sim.grid_velocity_x.to_numpy()[:-1, :]
            vy = self.sim.grid_velocity_y.to_numpy()[:, :-1]
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
        elif event.key == "1":
            self.scalar_field_to_render = "density"
        elif event.key == "2":
            self.scalar_field_to_render = "pressure"
        elif event.key == "3":
            self.scalar_field_to_render = "divergence"
        elif event.key == "4":
            self.scalar_field_to_render = "vorticity"
        elif event.key == "5":
            self.scalar_field_to_render = "grid_velocity_x"
        elif event.key == "6":
            self.scalar_field_to_render = "grid_velocity_y"
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


