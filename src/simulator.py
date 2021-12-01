# Based on code provided in Exercise 4

import taichi as ti
from MGPCGSolver import MGPCGSolver

# Note: all physical properties are in SI units (s for time, m for length, kg for mass, etc.)
global_params = {
    'mode' : 'flip',                             # pic, apic, flip
    'flip_weight' : 0.95,                       # FLIP * flip_weight + PIC * (1 - flip_weight)
    'dt' : 0.01,                                # Time step
    'g' : (0.0, 0.0, -9.8),                     # Body force
    'rho': 1000.0,                              # Density of the fluid
    'grid_size' : (64, 64, 64),                 # Grid size (integer)
    'cell_extent': 0.1,                        # Extent of a single cell. grid_extent equals to the product of grid_size and cell_extent

    'mac_cormack' : False,
    'gauss_seidel_max_iterations' : 1000,
    'gauss_seidel_min_accuracy' : 1e-5,
    'use_mgpcg' : False,
    'num_jacobi_iter' : 100,
    'damped_jacobi_weight' : 0.67,
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

        self.mode = get_param('mode')
        self.flip_weight = get_param('flip_weight')
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

        self.num_jacobi_iter = get_param('num_jacobi_iter')
        self.damped_jacobi_weight = get_param('damped_jacobi_weight')

        # simulation state
        self.cur_step = 0
        self.t = 0.0
        # self.r_wind = ti.Vector.field(1, ti.i32, shape=(1))

        self.init_fields()

        self.init_particles((16, 16, 0), (48, 48, 60))  # todo

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

        self.grid_velocity_x_last = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_velocity_y_last = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_velocity_z_last = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))
        # For renormalization after p2g
        self.grid_weight_x = ti.field(ti.f32, shape=(self.grid_size[0] + 1, self.grid_size[1], self.grid_size[2]))
        self.grid_weight_y = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1] + 1, self.grid_size[2]))
        self.grid_weight_z = ti.field(ti.f32, shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2] + 1))

        self.clear_grid()
        self.divergence = ti.field(ti.f32, shape=self.grid_size)
        self.new_pressure = ti.field(ti.f32, shape=self.grid_size)

    # todo: this is only a very naive particle init method:
    # it just select a portion of the grid and fill one particle for each cell
    def init_particles(self, range_min, range_max):
        range_min = ti.max(ti.Vector(range_min), 1)
        range_max = ti.min(ti.Vector(range_max), self.grid_size-1)

        # Number of particles
        range_size = range_max - range_min
        self.num_particles = range_size.x * range_size.y * range_size.z
        print(self.num_particles)

        # Particles
        self.particles_position = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particles_velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particles_affine_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_particles)

        self.init_particles_kernel(range_min[0], range_min[1], range_min[2], range_max[0], range_max[1], range_max[2])

        # for i in range(range_min[0], range_max[0]):
        #     for j in range(range_min[1], range_max[1]):
        #         for k in range(range_min[2], range_max[2]):
        #             self.cell_type[i, j, k] = FLUID

    @ti.kernel
    def init_particles_kernel(self, range_min_x:ti.i32, range_min_y:ti.i32,range_min_z:ti.i32,range_max_x:ti.i32,range_max_y:ti.i32,range_max_z:ti.i32):
        range_min = ti.Vector([range_min_x, range_min_y, range_min_z])
        range_max = ti.Vector([range_max_x, range_max_y, range_max_z])
        particle_init_size = range_max - range_min
        for p in self.particles_position:
            k = p % (range_max_z - range_min_z) + range_min_z
            j = (p // (range_max_z - range_min_z)) % (range_max_y - range_min_y) + range_min_y
            i = (p // ((range_max_z - range_min_z) * (range_max_y - range_min_y))) % (range_max_x - range_min_x) + range_min_x
            if self.cell_type[i, j, k] != SOLID:
                self.cell_type[i, j, k] = FLUID
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

        # Clear the grid for each step
        self.clear_grid()

        # print('Velocity after apply_force:', self.particles_velocity[self.num_particles//2])

        # Scatter properties (mainly velocity) from particle to grid
        self.p2g()
        # print('Velocity (grid) after p2g:', self.grid_velocity_z[self.grid_size.x//2, self.grid_size.y//2, self.grid_size.z//2],
        #       'weight:', self.grid_weight_z[self.grid_size.x//2, self.grid_size.y//2, self.grid_size.z//2]
        #       )

        if self.mode == 'flip':
            self.grid_velocity_x_last.copy_from(self.grid_velocity_x)
            self.grid_velocity_y_last.copy_from(self.grid_velocity_y)
            self.grid_velocity_z_last.copy_from(self.grid_velocity_z)

        # Apply body force
        self.apply_force()

        self.enforce_boundary_condition()

        self.compute_divergence()

        # Solve the poisson equation to get pressure
        self.solve_pressure()

        # Accelerate velocity using the solved pressure
        self.project_velocity()

        # Gather properties (mainly velocity) from grid to particle
        self.g2p()
        # print('Velocity after g2p:', self.particles_velocity[self.num_particles//2])


        # Advect particles
        self.advect_particles()

        # print('Position after advect_particles:', self.particles_position[self.num_particles//2])

        # Mark grid cell type as FLUID, AIR or SOLID (boundary)
        self.mark_cell_type()

    # Clear grid velocities and weights to 0
    def clear_grid(self):
        self.grid_velocity_x.fill(0.0)
        self.grid_velocity_y.fill(0.0)
        self.grid_velocity_z.fill(0.0)

        self.grid_weight_x.fill(0.0)
        self.grid_weight_y.fill(0.0)
        self.grid_weight_z.fill(0.0)

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
        # for p in self.particles_velocity:
        #     self.particles_velocity[p] += self.dt * self.g

        for i, j, k in self.grid_velocity_z:
            self.grid_velocity_z[i, j, k] -= 9.8 * self.dt

    @ti.kernel
    def p2g(self):
        for p in self.particles_position:
            xp = self.particles_position[p]
            vp = self.particles_velocity[p]
            cp = self.particles_affine_C[p]
            idx = xp/self.dx
            base = ti.cast(ti.floor(idx), dtype=ti.i32)
            frac = idx - base
            self.interp_grid(base, frac, vp, cp)

        for i, j, k in self.grid_velocity_x:
            v = self.grid_velocity_x[i, j, k]
            w = self.grid_weight_x[i,j,k]
            self.grid_velocity_x[i, j, k] = v / w if w > 0.0 else 0.0
        for i, j, k in self.grid_velocity_y:
            v = self.grid_velocity_y[i, j, k]
            w = self.grid_weight_y[i,j,k]
            self.grid_velocity_y[i, j, k] = v / w if w > 0.0 else 0.0
        for i, j, k in self.grid_velocity_z:
            v = self.grid_velocity_z[i, j, k]
            w = self.grid_weight_z[i,j,k]
            self.grid_velocity_z[i, j, k] = v / w if w > 0.0 else 0.0

    @ti.kernel
    def g2p(self):
        for p in self.particles_position:
            xp = self.particles_position[p]
            # vp = self.particles_velocity[p]
            idx = xp/self.dx
            base = ti.cast(ti.floor(idx), dtype=ti.i32)
            frac = idx - base
            # if p == 15:
            #     print('g2p:', xp, idx, base, frac)
            self.interp_particle(base, frac, p)

    # @ti.kernel
    def solve_pressure(self):
        if self.use_mgpcg:
            scale_A = self.dt / (self.rho * (self.dx ** 2))
            scale_b = 1 / self.dx

            self.mgpcg_solver.system_init(scale_A, scale_b)
            self.mgpcg_solver.solve(100)

            self.pressure.copy_from(self.mgpcg_solver.p)
        else:
            for i in range(self.num_jacobi_iter):
                self.jacobi_iter()
                self.pressure.copy_from(self.new_pressure)
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
            self.particles_position[p] += self.particles_velocity[p] * self.dt

        for p in self.particles_position:
            pos = self.particles_position[p]
            v = self.particles_velocity[p]

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
        for i, j, k in self.cell_type:
            if self.cell_type[i, j, k] == SOLID:
                self.grid_velocity_x[i, j, k] = 0.0
                self.grid_velocity_x[i + 1, j, k] = 0.0
                self.grid_velocity_y[i, j, k] = 0.0
                self.grid_velocity_y[i, j + 1, k] = 0.0
                self.grid_velocity_z[i, j, k] = 0.0
                self.grid_velocity_z[i, j, k + 1] = 0.0


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
    def interp_grid(self, base, frac, vp, cp):
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
                    if self.mode == 'apic':
                        dpos = ti.Vector([i-1, j-0.5, k-0.5]) - frac
                        self.grid_velocity_x[idx] += w * (cp @ dpos).x
                    self.grid_weight_x[idx] += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                for k in ti.static(range(3)):
                    w = w_center[i].x * w_side[j].y * w_center[k].z
                    idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                    self.grid_velocity_y[idx] += vp.y * w
                    if self.mode == 'apic':
                        dpos = ti.Vector([i-0.5, j-1, k-0.5]) - frac
                        self.grid_velocity_y[idx] += w * (cp @ dpos).y
                    self.grid_weight_y[idx] += w

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(4)):
                    w = w_center[i].x * w_center[j].y * w_side[k].z
                    idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                    # if idx[0] == self.grid_size.x // 2 and idx[1] == self.grid_size.y //2 and idx[2] == self.grid_size.z // 2:
                    #     print('weight p2g:', w_center[i].x, w_center[j].y, w_side[k].z)
                    self.grid_velocity_z[idx] += vp.z * w
                    if self.mode == 'apic':
                        dpos = ti.Vector([i-0.5, j-0.5, k-1]) - frac
                        self.grid_velocity_z[idx] += w * (cp @ dpos).z
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

        wx = 0.0
        wy = 0.0
        wz = 0.0
        vx = 0.0
        vy = 0.0
        vz = 0.0

        C_x = ti.Matrix.zero(ti.f32, 3)
        C_y = ti.Matrix.zero(ti.f32, 3)
        C_z = ti.Matrix.zero(ti.f32, 3)

        vx_d = 0.0
        vy_d = 0.0
        vz_d = 0.0

        for i in ti.static(range(4)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    w = w_side[i].x * w_center[j].y * w_center[k].z
                    idx = (idx_side[i].x, idx_center[j].y, idx_center[k].z)
                    vtemp = self.grid_velocity_x[idx] * w
                    vx += vtemp
                    if self.mode == 'flip':
                        vx_d += vtemp - self.grid_velocity_x_last[idx] * w
                    if self.mode == 'apic':
                        dpos = ti.Vector([i-1, j-0.5, k-0.5]) - frac
                        C_x += 4 * vtemp * dpos  / self.grid_size.x
                    wx += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                for k in ti.static(range(3)):
                    w = w_center[i].x * w_side[j].y * w_center[k].z
                    idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                    vtemp = self.grid_velocity_y[idx] * w
                    vy += vtemp
                    if self.mode == 'flip':
                        vy_d += vtemp - self.grid_velocity_y_last[idx] * w
                    if self.mode == 'apic':
                        dpos = ti.Vector([i-0.5, j-1, k-0.5]) - frac
                        C_y += 4 * vtemp * dpos / self.grid_size.y
                    wy += w

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(4)):
                    w = w_center[i].x * w_center[j].y * w_side[k].z
                    idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                    vtemp = self.grid_velocity_z[idx] * w
                    vz += vtemp
                    if self.mode == 'flip':
                        vz_d += vtemp - self.grid_velocity_z_last[idx] * w
                    if self.mode == 'apic':
                        dpos = ti.Vector([i-0.5, j-0.5, k-1]) - frac
                        C_z += 4 * vtemp * dpos / self.grid_size.z
                    wz += w
                    # if p == 15 and i == 0 and j == 0 and k == 1:
                    #     print('interp_particle000:', w_center[i].x, w_center[j].y, w_side[k].z, idx)
        # if p == 15:
        #     print('interp_particle:', base, frac, velocity, weight, w_side[1])
        # The weight will never be 0
        if self.mode == 'flip':
            vp = self.particles_velocity[p]
            vxp = vp.x + vx_d/wx
            vyp = vp.y + vy_d/wy
            vzp = vp.z + vz_d/wz
            self.particles_velocity[p] = ti.Vector([vxp * self.flip_weight + vx/wx * (1 - self.flip_weight),
                                                    vyp * self.flip_weight + vy/wy * (1 - self.flip_weight),
                                                    vzp * self.flip_weight + vz/wz * (1 - self.flip_weight)])
        else:
            self.particles_velocity[p] = ti.Vector([vx/wx, vy/wy, vz/wz])
        self.particles_affine_C[p] = ti.Matrix.rows([C_x/wx, C_y/wy, C_z/wz])

    @ti.kernel
    def compute_divergence(self):
        for i, j, k in self.divergence:
            if not self.is_solid(i, j, k):
                div = (self.grid_velocity_x[i + 1, j, k] - self.grid_velocity_x[i, j, k])
                div += (self.grid_velocity_y[i, j + 1, k] - self.grid_velocity_y[i, j, k])
                div += (self.grid_velocity_z[i, j, k + 1] - self.grid_velocity_z[i, j, k])

                if self.is_solid(i-1, j, k):
                    div += self.grid_velocity_x[i, j, k]
                if self.is_solid(i+1, j, k):
                    div -= self.grid_velocity_x[i + 1, j, k]
                if self.is_solid(i, j-1, k):
                    div += self.grid_velocity_y[i, j, k]
                if self.is_solid(i, j+1, k):
                    div -= self.grid_velocity_y[i, j + 1, k]
                if self.is_solid(i, j, k-1):
                    div += self.grid_velocity_z[i, j, k]
                if self.is_solid(i, j, k+1):
                    div -= self.grid_velocity_z[i, j, k + 1]

                self.divergence[i, j, k] = div
            else:
                self.divergence[i, j, k] = 0.0
            self.divergence[i, j, k] /= self.dx

    @ti.kernel
    def jacobi_iter(self):
        for i, j, k in self.pressure:
            if self.is_fluid(i, j, k):
                div = self.divergence[i, j, k]

                p_x1 = self.pressure[i - 1, j, k]
                p_x2 = self.pressure[i + 1, j, k]
                p_y1 = self.pressure[i, j - 1, k]
                p_y2 = self.pressure[i, j + 1, k]
                p_z1 = self.pressure[i, j, k - 1]
                p_z2 = self.pressure[i, j, k + 1]
                n = 6
                if self.is_solid(i-1, j, k):
                    p_x1 = 0.0
                    n -= 1
                if self.is_solid(i+1, j, k):
                    p_x2 = 0.0
                    n -= 1
                if self.is_solid(i, j-1, k):
                    p_y1 = 0.0
                    n -= 1
                if self.is_solid(i, j+1, k):
                    p_y2 = 0.0
                    n -= 1
                if self.is_solid(i, j, k-1):
                    p_z1 = 0.0
                    n -= 1
                if self.is_solid(i, j, k+1):
                    p_z2 = 0.0
                    n -= 1

                self.new_pressure[i, j, k] = (1 - self.damped_jacobi_weight) * self.pressure[i, j, k] +\
                                             self.damped_jacobi_weight * (p_x1 + p_x2 + p_y1 + p_y2 + p_z1 + p_z2 - div * self.rho / self.dt * (self.dx ** 2)) / n
            else:
                self.new_pressure[i, j, k] = 0.0





