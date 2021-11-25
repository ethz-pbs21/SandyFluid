# Based on code provided in Exercise 4

import taichi as ti
import numpy as np

# Note: all physical properties are in SI units (s for time, m for length, kg for mass, etc.)
params = {
    'dt' : 0.045,                               # Time step
    'g' : (0.0, 0.0, -9.8),                     # Body force
    'rho': 1000.0,                              # Density of the fluid
    'grid_size' : (64, 64, 64),                 # Grid size (integer)
    'cell_extent': 0.01,                        # Extent of a single cell. grid_extent equals to the product of grid_size and cell_extent

    'mac_cormack' : False,
    'gauss_seidel_max_iterations' : 1000,
    'gauss_seidel_min_accuracy' : 1e-5,
    'use_mgpcg' : True,
}

FLUID = 0
AIR = 1
SOLID = 2

@ti.kernel
def abs_vector(f1: ti.template(), f2: ti.template(), fout: ti.template()):
    for x, y in ti.ndrange(*fout.shape):
        fout[x, y] = ti.sqrt(f1[x, y] ** 2 + f2[x, y] ** 2)


@ti.data_oriented
class Simulator(object):
    def __init__(self, params : dict):
        def get_param(key:str, default_val=None):
            return params[key] if key in params else default_val

        # Number of particles
        self.num_particles = # todo

        # Time step
        self.dt = get_param('dt')
        # Body force (gravity)
        self.g = np.array(get_param('g'), dtype=np.float32)

        self.paused = True

        # parameters that are fixed (changing these after starting the simulatin will not have an effect!)
        self.grid_size = np.array(get_param('grid_size'), dtype=np.int32)
        self.inv_grid_size = 1.0 / self.grid_size.astype(np.float32)
        self.cell_extent = get_param('cell_extent')
        self.grid_extent = self.grid_size.astype(np.float32) * self.cell_extent
        self.dx = 1.0 / self.grid_size[0] # todo

        self.rho = get_param('rho')
        

        # simulation state
        self.cur_step = 0
        self.t = 0.0
        # self.r_wind = ti.Vector.field(1, ti.i32, shape=(1))

        # pressure solver type
        self.use_mgpcg = get_param('use_mgpcg')

        self.init_fields()
        self.init_particles((16, 16, 32), (48, 48, 64))  # todo

        self.reset()

    def init_fields(self):
        # MAC grid
        self.pressure = ti.field(ti.f32, shape=self.grid_size)
        # mark each grid as FLUID = 0, AIR = 1 or SOLID = 2
        self.celltype = ti.field(ti.i32, shape=self.grid_size)

        self.grid_velocity_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_velocity_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_velocity_z = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))
        # For renormalization after p2g
        self.grid_weight_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_weight_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_weight_z = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))

        # self.divergence = ti.field(ti.f32, shape=self.grid_size)

    # todo: this is only a very naive particle init method: just select a portion of the grid and fill one particle for each cell

    def init_particles(self, range_min, range_max):
        range_min = np.max(np.array(range_min), 0)
        range_max = np.min(np.arrag(range_max), self.grid_size)
        self.num_particles = (range_max-range_min).prod()

        # Particles
        self.particles_position = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particles_velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)

        self.init_particles_kernel(ti.Vector(range_min), ti.Vector(range_max))


    @ti.kernel
    def init_particles_kernel(self, range_min, range_max):
        particle_init_size = range_max - range_min
        for p in self.particles_position:
            k = p % (particle_init_size.z * particle_init_size.y) + range_min.z
            j = (p // particle_init_size.z) % particle_init_size.y + range_min.y
            i = p / (particle_init_size.z * particle_init_size.y) + range_min.x
            self.particles_position[p] = (ti.Vector([i,j,k]) + 0.5) * self.cell_extent

    def init_solver(self):
        # init pressure solver
        if self.use_mgpcg:
            self.mgpcg_solver = utils.MGPCGSolver(self.grid_size,
                                            self.grid_velocity_x,
                                            self.grid_velocity_y,
                                            self.grid_velocity_z,
                                            self.cell_type)
        # else:
        #     self.gauss_seidel_max_iterations = int(gauss_seidel_max_iterations)
        #     self.gauss_seidel_min_accuracy = gauss_seidel_min_accuracy


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

        # Mark grid cell type as FLUID, AIR or SOLID (boundary)
        self.mark_cell_type()

    # Helper
    @ti.func
    def is_valid(self, i, j, k):
        return 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1] and 0 <= k < self.grid_size[2]

    @ti.func
    def is_fluid(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == FLUID

    @ti.func
    def is_air(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == AIR

    @ti.func
    def is_solid(self, i, j, k):
        return self.is_valid(i, j, k) and self.cell_type[i, j, k] == SOLID

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
            self.grid_velocity_x[i, j, k] /= self.grid_weight_x[i,j,k]
        for i, j, k in self.grid_velocity_y:
            self.grid_velocity_y[i, j, k] /= self.grid_weight_y[i,j,k]
        for i, j, k in self.grid_velocity_x:
            self.grid_velocity_z[i, j, k] /= self.grid_weight_z[i,j,k]

    @ti.kernel
    def g2p():
        for p in self.particles_position:
            xp = self.particles_position[p]
            vp = self.particles_velocity[p]
            base = ti.floor(xp)
            frac = xp - base
            self.interp_particle(base, frac, p)

    @ti.kernel
    def solve_pressure(
        self,
        dt: ti.f32,
        gauss_seidel_max_iterations: ti.i32,
        gauss_seidel_min_accuracy: ti.f32,
    ):
        if self.use_mgpcg:
            scale_A = dt / (self.rho * self.dx ** 2)
            scale_b = 1 / self.dx

            self.mgpcg_solver.system_init(scale_A, scale_b)
            self.mgpcg_solver.solve(500)

            self.pressure.copy_from(self.mgpcg_solver.p)
        # else:
        #     # set Neumann boundary conditons
        #     for y in range(0, self.grid_size[1]):
        #         self.grid_velocity_x[0, y] = self.grid_velocity_x[2, y]
        #         self.grid_velocity_x[-1, y] = self.grid_velocity_x[-3, y]
        #     for x in range(0, self.grid_size[0]):
        #         self.grid_velocity_y[x, 0] = self.grid_velocity_y[x, 2]
        #         self.grid_velocity_y[x, -1] = self.grid_velocity_y[x, -3]
        #
        #     # set tangential velocities along edges to zero
        #     for x in range(0, self.grid_size[0] + 1):
        #         self.grid_velocity_x[x, 0] = 0
        #         self.grid_velocity_x[x, -1] = 0
        #     for y in range(0, self.grid_size[1] + 1):
        #         self.grid_velocity_y[0, y] = 0
        #         self.grid_velocity_y[-1, y] = 0
        #
        #     # compute divergence
        #     self.compute_divergence()
        #
        #     # copy border pressures
        #     for y in range(0, self.grid_size[1] + 1):
        #         self.pressure[0, y] = self.pressure[1, y]
        #         self.pressure[-1, y] = self.pressure[-2, y]
        #     for x in range(0, self.grid_size[0] + 1):
        #         self.pressure[x, 0] = self.pressure[x, 1]
        #         self.pressure[x, -1] = self.pressure[x, -2]
        #
        #     # solve possiion equation
        #     gauss_seidel_poisson_solver(
        #         self.pressure,
        #         self.divergence,
        #         dt,
        #         self.dx,
        #         gauss_seidel_max_iterations,
        #         gauss_seidel_min_accuracy,
        #         self.rho,
        #     )

    @ti.kernel
    def project_velocity(self, dt: ti.f32):
        scale = dt / (self.rho * self.dx)
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.is_fluid(i - 1, j, k) or self.is_fluid(i, j, k):
                if self.is_solid(i - 1, j, k) or self.is_solid(i, j, k):
                    self.grid_velocity_x[i, j, k] = 0 # u_solid = 0
                else:
                    self.grid_velocity_x[i, j, k] -= scale * (self.pressure[i, j, k] - self.pressure[i - 1, j, k])

            if self.is_fluid(i, j - 1, k) or self.is_fluid(i, j, k):
                if self.is_solid(i, j - 1, k) or self.is_solid(i, j, k):
                    self.grid_velocity_y[i, j, k] = 0
                else:
                    self.grid_velocity_y[i, j, k] -= scale * (self.pressure[i, j, k] - self.pressure[i, j - 1, k])

            if self.is_fluid(i, j, k - 1) or self.is_fluid(i, j, k):
                if self.is_solid(i, j, k - 1) or self.is_solid(i, j, k):
                    self.grid_velocity_z[i, j, k] = 0
                else:
                    self.grid_velocity_z[i, j, k] -= scale * (self.pressure[i, j, k] - self.pressure[i, j, k - 1])


    @ti.kernel
    def enforce_boundary_condition(self):
        pass

    @ti.kernel
    def advect_particles(self):
        pass

    @ti.kernel
    def mark_cell_type(self):
        for i, j, k in self.cell_type:
            if not self.is_solid(i, j, k):
                self.cell_type[i, j, k] = AIR

        for p in self.particles_position:
            xp = self.particles_position[p]
            idx = ti.cast(ti.floor(xp / self.dx), ti.i32)

            if not self.is_solid(idx[0], idx[1], idx[2]):
                self.cell_type[idx] = FLUID

    # ###########################################################
    # Funcs called by kernels

    # Spline functions for interpolation
    # Input x should be non-negative (abs)

    # Quadratic B-spline
    # 0.75-x^2,         |x| in [0, 0.5)
    # 0.5*(1.5-|x|)^2,  |x| in [0.5, 1.5)
    # 0,                |x| in [1.5, inf)
    @ti.func
    def quadratic_kernel(x : ti.Matrix):
        w = x.copy()
        for i, xi in enumerate(x):
            if xi < 0.5:
                w[i] = 0.75 - xi**2
            elif xi < 1.5:
                w[i] = 0.5 * (1.5 - xi)**2
            else:
                w[i] = 0.0
        return w

    
    @ti.func
    def interp_grid(self, base : ti.Matrix, frac : ti.Matrix, vp : ti.Matrix):
        
        # Quadratic
        # todo: try other kernels (linear, cubic, ...)

        # Index on sides
        idx_side = [base-1, base, base+1, base+2]
        # Weight on sides
        w_side = [self.quadratic_kernel(1.0+frac), self.quadratic_kernel(frac), self.quadratic_kernel(1.0-frac), self.quadratic_kernel(2.0-frac)]
        # Index on centers
        idx_center = [base-1, base, base+1]
        # Weight on centers
        w_center = [self.quadratic_kernel(0.5+frac), self.quadratic_kernel(ti.abs(0.5-frac)), self.quadratic_kernel(1.5-frac)]


        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    w = w_side[i].x * w_center[j].y * w_center[k].z
                    idx = (idx_side[i].x, idx_center[j].y, idx_center[k].z)
                    self.grid_velocity_x[idx] += vp.x * w
                    self.grid_weight_x[idx] += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                for k in ti.static(range(3)):
                    w = w_center[i].x * w_side[j].y * w_center[k].z
                    idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                    self.grid_velocity_y[idx] += vp.y * w
                    self.grid_weight_y[idx] += w
        
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(4)):
                    w = w_center[i].x * w_center[j].y * w_side[k].z
                    idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                    self.grid_velocity_z[idx] += vp.z * w
                    self.grid_weight_z[idx] += w

    @ti.func
    def interp_particle(self, base : ti.Matrix, frac : ti.Matrix, p):
        pass

    @ti.func
    def compute_divergence(self):
        for i, j, k in ti.ndrange(
            (1, self.grid_size[0] - 1), (1, self.grid_size[1] - 1), (1, self.grid_size[2] - 1)
        ):
            dudx = (self.grid_velocity_x[x + 1, y] - self.grid_velocity_x[x, y]) / self.dx
            dudy = (self.grid_velocity_y[x, y + 1] - self.grid_velocity_y[x, y]) / self.dx
            self.divergence[x, y] = dudx + dudy




