# Based on code provided in Exercise 4

import taichi as ti
from MGPCGSolver import MGPCGSolver

# Note: all physical properties are in SI units (s for time, m for length, kg for mass, etc.)
global_params = {
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
    def __init__(self, params : dict = global_params):
        def get_param(key:str, default_val=None):
            return params[key] if key in params else default_val


        # Time step
        self.dt = get_param('dt')
        # Body force (gravity)
        self.g = ti.Vector(get_param('g'), dt=ti.f32)  # todo: Datatype decl seems to be noneffective here...

        self.paused = True

        # parameters that are fixed (changing these after starting the simulatin will not have an effect!)
        self.grid_size = ti.Vector(get_param('grid_size'), dt=ti.i32)
        # self.inv_grid_size = 1.0 / self.grid_size
        self.cell_extent = get_param('cell_extent')
        self.grid_extent = self.grid_size * self.cell_extent
        # self.dx = 1.0 / self.grid_size[0] # todo
        self.dx = self.cell_extent

        self.rho = get_param('rho')
        

        # simulation state
        self.cur_step = 0
        self.t = 0.0
        # self.r_wind = ti.Vector.field(1, ti.i32, shape=(1))

        self.init_fields()

        self.init_particles((16, 16, 32), (48, 48, 64))  # todo

        # pressure solver type
        self.use_mgpcg = get_param('use_mgpcg')
        self.init_solver()

        self.reset()

    def init_fields(self):
        # MAC grid
        self.pressure = ti.field(ti.f32, shape=self.grid_size)

        # mark each grid as FLUID = 0, AIR = 1 or SOLID = 2
        self.cell_type = ti.field(ti.i32, shape=self.grid_size)
        self.cell_type.fill(AIR)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.cell_type[i, j, 0] = SOLID
                self.cell_type[i, j, self.grid_size[2]-1] = SOLID

        for j in range(self.grid_size[1]):
            for k in range(self.grid_size[2]):
                self.cell_type[0, j, k] = SOLID
                self.cell_type[self.grid_size[0]-1, j, k] = SOLID

        for i in range(self.grid_size[0]):
            for k in range(self.grid_size[2]):
                self.cell_type[i, 0, k] = SOLID
                self.cell_type[i, self.grid_size[1]-1, k] = SOLID

        self.grid_velocity_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_velocity_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_velocity_z = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))
        # For renormalization after p2g
        self.grid_weight_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_weight_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_weight_z = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))

        # self.divergence = ti.field(ti.f32, shape=self.grid_size)

    # todo: this is only a very naive particle init method:
    # it just select a portion of the grid and fill one particle for each cell
    def init_particles(self, range_min, range_max):
        range_min = ti.max(ti.Vector(range_min), 0)
        range_max = ti.min(ti.Vector(range_max), self.grid_size)

        # Number of particles
        range_size = range_max - range_min
        self.num_particles = range_size.x * range_size.y * range_size.z

        # Particles
        self.particles_position = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particles_velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)

        self.init_particles_kernel(range_min[0], range_min[1], range_min[2], range_max[0], range_max[1], range_max[2])

        for i in range(range_max[0] - range_min[0]):
            for j in range(range_max[1] - range_min[1]):
                for k in range(range_max[2] - range_min[2]):
                    self.cell_type[i+range_min[0], j+range_min[1], k+range_min[2]] = FLUID

    @ti.kernel
    def init_particles_kernel(self, range_min_x:ti.i32, range_min_y:ti.i32,range_min_z:ti.i32,range_max_x:ti.i32,range_max_y:ti.i32,range_max_z:ti.i32):
        range_min = ti.Vector([range_min_x, range_min_y, range_min_z])
        range_max = ti.Vector([range_max_x, range_max_y, range_max_z])
        particle_init_size = range_max - range_min
        for p in self.particles_position:
            k = p % (particle_init_size.z * particle_init_size.y) + range_min.z
            j = (p // particle_init_size.z) % particle_init_size.y + range_min.y
            i = p / (particle_init_size.z * particle_init_size.y) + range_min.x
            self.particles_position[p] = (ti.Vector([i,j,k]) + 0.5) * self.cell_extent

    def init_solver(self):
        # init pressure solver
        if self.use_mgpcg:
            self.mgpcg_solver = MGPCGSolver(self.grid_size,
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


    def step(self):
        self.cur_step += 1
        self.t += self.dt

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
            base = ti.cast(ti.floor(xp/self.dx), dtype=ti.int32)
            frac = xp - base
            self.interp_grid(base, frac, vp)

        for i, j, k in self.grid_velocity_x:
            self.grid_velocity_x[i, j, k] /= self.grid_weight_x[i,j,k]
        for i, j, k in self.grid_velocity_y:
            self.grid_velocity_y[i, j, k] /= self.grid_weight_y[i,j,k]
        for i, j, k in self.grid_velocity_x:
            self.grid_velocity_z[i, j, k] /= self.grid_weight_z[i,j,k]

    @ti.kernel
    def g2p(self):
        for p in self.particles_position:
            xp = self.particles_position[p]
            vp = self.particles_velocity[p]
            base = ti.cast(ti.floor(xp/self.dx), dtype=ti.int32)
            frac = xp - base
            self.interp_particle(base, frac, p)

    # @ti.kernel
    def solve_pressure(self):
        if self.use_mgpcg:
            scale_A = self.dt / (self.rho * self.dx ** 2)
            scale_b = 1 / self.dx

            self.mgpcg_solver.system_init(scale_A, scale_b)
            self.mgpcg_solver.solve(100)

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
    def project_velocity(self):
        scale = self.dt / (self.rho * self.dx)
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
    def advect_particles(self):
        # Forward Euler
        # todo: RK2
        for p in self.particles_position:
            pos = self.particles_position[p]
            v = self.particles_velocity[p]
            pos += v * self.dt
            # todo: boundary condition, for velocity
            # self.particles_position[p] = ti.min(ti.max(self.particles_position[p], 0), self.grid_extent)

            for i in ti.static(range(3)):
                if pos[i] <= self.cell_extent:
                    pos[i] = self.cell_extent
                    v[i] = 0
                if pos[i] >= self.grid_extent[i]-self.cell_extent:
                    pos[i] = self.grid_extent[i]-self.cell_extent
                    v[i] = 0

            self.particles_position[p] = pos
            self.particles_velocity[p] = v

    @ti.kernel
    def enforce_boundary_condition(self):
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
    def quadratic_kernel(self, x):
        w = ti.Vector([0.0 for _ in ti.static(range(3))])
        for i in ti.static(range(3)):  # todo: maybe we should not assume x.n==3 here
            if x[i] < 0.5:
                w[i] = 0.75 - x[i]**2
            elif x[i] < 1.5:
                w[i] = 0.5 * (1.5 - x[i])**2
            else:
                w[i] = 0.0
        return w

    
    @ti.func
    def interp_grid(self, base, frac, vp):
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
    def interp_particle(self, base, frac, p):
        # Index on sides
        idx_side = [base-1, base, base+1, base+2]
        # Weight on sides
        w_side = [self.quadratic_kernel(1.0+frac), self.quadratic_kernel(frac), self.quadratic_kernel(1.0-frac), self.quadratic_kernel(2.0-frac)]
        # Index on centers
        idx_center = [base-1, base, base+1]
        # Weight on centers
        w_center = [self.quadratic_kernel(0.5+frac), self.quadratic_kernel(ti.abs(0.5-frac)), self.quadratic_kernel(1.5-frac)]

        weight = 0.0
        velocity = ti.Vector([0.0, 0.0, 0.0])

        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    w = w_side[i].x * w_center[j].y * w_center[k].z
                    idx = (idx_side[i].x, idx_center[j].y, idx_center[k].z)
                    velocity += self.grid_velocity_x[idx] * w
                    weight += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                for k in ti.static(range(3)):
                    w = w_center[i].x * w_side[j].y * w_center[k].z
                    idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                    velocity += self.grid_velocity_y[idx] * w
                    weight += w

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(4)):
                    w = w_center[i].x * w_center[j].y * w_side[k].z
                    idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                    velocity += self.grid_velocity_z[idx] * w
                    weight += w

        self.particles_velocity[p] = velocity / weight

    # @ti.func
    # def compute_divergence(self):
    #     for i, j, k in ti.ndrange(
    #         (1, self.grid_size[0] - 1), (1, self.grid_size[1] - 1), (1, self.grid_size[2] - 1)
    #     ):
    #         dudx = (self.grid_velocity_x[x + 1, y] - self.grid_velocity_x[x, y]) / self.dx
    #         dudy = (self.grid_velocity_y[x, y + 1] - self.grid_velocity_y[x, y]) / self.dx
    #         self.divergence[x, y] = dudx + dudy




