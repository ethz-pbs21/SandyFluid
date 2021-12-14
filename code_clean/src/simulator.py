# Based on code provided in Exercise 4

import taichi as ti
import mcubes
import numpy as np
import scipy.spatial
import cc3d
import os
import time
# Note: all physical properties are in SI units (s for time, m for length, kg for mass, etc.)
global_params = {
    'mode' : 'apic',                            # pic, apic, flip
    'flip_weight' : 0.99,                       # FLIP * flip_weight + PIC * (1 - flip_weight)
    'dt' : 0.01,                                # Time step
    'g' : (0.0, 0.0, -9.8),                     # Body force
    'rho': 1000.0,                              # Density of the fluid
    'grid_size' : (64, 64, 64),                 # Grid size (integer)
    'cell_extent': 0.1,                         # Extent of a single cell. grid_extent equals to the product of grid_size and cell_extent

    'reconstruct_resolution': (100, 100, 100),  # Mesh surface reconstruction grid resolution
    'reconstruct_threshold' : 0.75,             # Threshold of the metaball scalar fields
    'reconstruct_radius' : 0.1,                 # Radius of the metaball

    'num_jacobi_iter' : 100,                    # Number of iterations for pressure solving
    'damped_jacobi_weight' : 1.0,               # Damping weighte in damped jacobi

    'simulate_sand': False,                     # Simulate sand or water
    'sand_dt': 0.001,                           # Time step for simulating sand

    'scene_init': 0,                            # Choose from 0, 1, 2 to init the particle positions differently
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
        self.simulate_sand = get_param('simulate_sand')
        # Time step
        self.dt = get_param('dt')
        if self.simulate_sand:
            self.dt = get_param('sand_dt')
        # Body force (gravity)
        self.g = ti.Vector(get_param('g'), dt=ti.f32)

        self.paused = True

        # parameters that are fixed (changing these after starting the simulatin will not have an effect!)
        self.grid_size = ti.Vector(get_param('grid_size'), dt=ti.i32)
        self.cell_extent = get_param('cell_extent')
        self.grid_extent = self.grid_size * self.cell_extent
        self.dx = self.cell_extent

        self.reconstruct_resolution = get_param('reconstruct_resolution')
        self.reconstruct_threshold = get_param('reconstruct_threshold')
        self.reconstruct_radius = get_param('reconstruct_radius')

        self.rho = get_param('rho')

        self.num_jacobi_iter = get_param('num_jacobi_iter')
        self.damped_jacobi_weight = get_param('damped_jacobi_weight')

        # simulation state
        self.cur_step = 0
        self.t = 0.0

        # friction coefficient
        self.mu = 0.6
        # boundary friction coefficient
        self.b_mu  = 0.8

        self.init_fields()

        self.scene_init = get_param('scene_init')
        if self.scene_init == 0:
            self.init_particles((16, 16, 0), (48, 48, 40))
        elif self.scene_init == 1:
            self.init_particles((16, 16, 12), (48, 48, 56))
        elif self.scene_init == 2:
            self.init_particles((0, 0, 0), (30, 30, 40))

        self.datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.proj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.result_dir = os.path.join(self.proj_dir, 'results', self.datetime)

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

        self.sand_state = ti.field(ti.i32, shape=self.grid_size)
        self.clear_grid()
        self.divergence = ti.field(ti.f32, shape=self.grid_size)
        self.new_pressure = ti.field(ti.f32, shape=self.grid_size)

        self.stress = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.grid_size)
        self.strain_rate = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.grid_size)
        self.group_label = ti.field(ti.i16, shape=self.grid_size)

    def init_particles(self, range_min, range_max):
        range_min = ti.max(ti.Vector(range_min), 1)
        range_max = ti.min(ti.Vector(range_max), self.grid_size-1)

        # Number of particles
        range_size = range_max - range_min
        self.num_particles = range_size.x * range_size.y * range_size.z
        print(self.num_particles)

        # Particles
        self.particles_position = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles * 4)
        self.particles_velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles * 4)
        self.particles_affine_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_particles * 4)

        self.init_particles_kernel(range_min[0], range_min[1], range_min[2], range_max[0], range_max[1], range_max[2])

    @ti.kernel
    def init_particles_kernel(self, range_min_x:ti.i32, range_min_y:ti.i32,range_min_z:ti.i32,range_max_x:ti.i32,range_max_y:ti.i32,range_max_z:ti.i32):
        range_min = ti.Vector([range_min_x, range_min_y, range_min_z])
        range_max = ti.Vector([range_max_x, range_max_y, range_max_z])
        particle_init_size = range_max - range_min
        for p in self.particles_position:
            p1 = p // 4
            k = p1 % (range_max_z - range_min_z) + range_min_z
            j = (p1 // (range_max_z - range_min_z)) % (range_max_y - range_min_y) + range_min_y
            i = (p1 // ((range_max_z - range_min_z) * (range_max_y - range_min_y))) % (range_max_x - range_min_x) + range_min_x
            if self.cell_type[i, j, k] != SOLID:
                self.cell_type[i, j, k] = FLUID
            self.particles_position[p] = (ti.Vector([i,j,k]) + ti.Vector([ti.random(), ti.random(), ti.random()])) * self.cell_extent

    def reset(self):
        # Reset simulation state
        self.cur_step = 0
        self.t = 0.0

    def metaball_scalar_field(self, resolution):
        dx = self.grid_extent[0] / resolution[0]
        dy = self.grid_extent[1] / resolution[1]
        dz = self.grid_extent[2] / resolution[2]

        radius = self.reconstruct_radius

        particle_pos = self.particles_position.to_numpy()
        kdtree = scipy.spatial.KDTree(particle_pos)

        # We are picking scalar values at vertices of a cell rather than the center
        # Also, step out one cell's distance to completely include the simulation region
        f_shape = (resolution[0]+3, resolution[1]+3, resolution[2]+3)
        i, j, k = np.mgrid[:f_shape[0], :f_shape[1], :f_shape[2]] - 1

        x = i * dx
        y = j * dy
        z = k * dz
        field_pos = np.stack((x, y, z), axis=3)
        particle_indices = kdtree.query_ball_point(field_pos, radius, workers=-1)

        f = np.zeros(f_shape)

        def metaball_eval(p1, p2s):
            if p2s is None:
                return 0.0
            r = np.clip(np.linalg.norm(p1 - p2s, axis=1) / radius, 0.0, 1.0)
            return (1.0 - r ** 3 * (r * (r * 6.0 - 15.0) + 10.0)).sum()

        for ii in range(resolution[0]):
            for jj in range(resolution[1]):
                for kk in range(resolution[2]):
                    idx = (ii, jj, kk)
                    p1 = np.array([x[idx], y[idx], z[idx]])
                    p2s = np.array([particle_pos[p2_idx] for p2_idx in particle_indices[idx]]) if particle_indices[idx] else None
                    f[idx] = metaball_eval(p1, p2s)
        return f


    def reconstruct_mesh(self, filename=None):
        f = self.metaball_scalar_field(self.reconstruct_resolution)
        vertices, triangles = mcubes.marching_cubes(f, self.reconstruct_threshold)  # Threshold is picked arbitrarily
        os.makedirs(self.result_dir, exist_ok=True)
        if filename is None:
            filename = '{0}_{1:04d}_{2}s.obj'.format(self.mode, self.cur_step, round(self.t, 3))
        path = os.path.join(self.result_dir, filename)
        mcubes.export_obj(vertices, triangles, path)


    def step(self):
        self.cur_step += 1
        self.t += self.dt

        # Clear the grid for each step
        self.clear_grid()

        # Scatter properties (mainly velocity) from particle to grid
        self.p2g()

        # Clear solid boundary velocity
        self.enforce_boundary_condition()

        if self.mode == 'flip':
            self.grid_velocity_x_last.copy_from(self.grid_velocity_x)
            self.grid_velocity_y_last.copy_from(self.grid_velocity_y)
            self.grid_velocity_z_last.copy_from(self.grid_velocity_z)

        # Apply body force
        self.apply_force()

        # Compute velocity divergence
        self.compute_divergence()

        # Solve the poisson equation to get pressure
        self.solve_pressure()

        # Accelerate velocity using the solved pressure
        self.project_velocity()

        if self.simulate_sand:
            # Compute strain rate tensor for each grid cell
            self.compute_stress()

            # Decide yield condition
            self.friction_condition()

            # Update velocity for rigidly moving grids
            self.update_sand_rigid_group()

            # Update velocity for flowing grids
            self.update_sand_flowing()

        # Gather properties (mainly velocity) from grid to particle
        self.g2p()

        # Advect particles
        self.advect_particles()

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

        self.sand_state.fill(0)

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
            if k > 1:
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
            idx = xp/self.dx
            base = ti.cast(ti.floor(idx), dtype=ti.i32)
            frac = idx - base

            self.interp_particle(base, frac, p)

    def solve_pressure(self):
        for i in range(self.num_jacobi_iter):
            self.jacobi_iter()
            self.pressure.copy_from(self.new_pressure)

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
        for p in self.particles_position:
            self.particles_position[p] += self.particles_velocity[p] * self.dt

        for p in self.particles_position:
            pos = self.particles_position[p]
            v = self.particles_velocity[p]

            if not self.simulate_sand:
                for i in ti.static(range(3)):
                    if pos[i] <= self.cell_extent:
                        pos[i] = self.cell_extent
                        v[i] = 0
                    if pos[i] >= self.grid_extent[i]-self.cell_extent:
                        pos[i] = self.grid_extent[i]-self.cell_extent
                        v[i] = 0
            else:
                for i in ti.static(range(3)):
                    if pos[i] <= self.cell_extent:
                        pos[i] = self.cell_extent
                    if pos[i] >= self.grid_extent[i]-self.cell_extent:
                        pos[i] = self.grid_extent[i]-self.cell_extent

                if pos[0] <= self.cell_extent * 2 or pos[0] >= self.grid_extent[0]-2 * self.cell_extent:
                    vn = v[0]
                    v[0] = 0.0
                    vt = ti.Vector([0.0, v[1], v[2]], dt=ti.f32)
                    vt = ti.max(0, 1 - self.b_mu * ti.abs(vn) / vt.norm()) * vt
                    v[1] = vt[1]
                    v[2] = vt[2]

                if pos[1] <= self.cell_extent * 2 or pos[1] >= self.grid_extent[1]-2 * self.cell_extent:
                    vn = v[1]
                    v[1] = 0.0
                    vt = ti.Vector([v[0], 0.0, v[2]], dt=ti.f32)
                    vt = ti.max(0, 1 - self.b_mu * ti.abs(vn) / vt.norm()) * vt
                    v[0] = vt[0]
                    v[2] = vt[2]

                if pos[2] <= self.cell_extent * 2 or pos[2] >= self.grid_extent[2]-2 * self.cell_extent:
                    vn = v[2]
                    v[2] = 0.0
                    vt = ti.Vector([v[0], v[1], 0.0], dt=ti.f32)
                    vt = ti.max(0, 1 - self.b_mu * ti.abs(vn) / vt.norm()) * vt
                    v[1] = vt[1]
                    v[0] = vt[0]

            self.particles_position[p] = pos
            self.particles_velocity[p] = v

    @ti.kernel
    def enforce_boundary_condition(self):
        for i, j in ti.ndrange(self.grid_size[0], self.grid_size[1]):
            self.grid_velocity_z[i, j, 0] = 0
            self.grid_velocity_z[i, j, 1] = 0
            self.grid_velocity_z[i, j, self.grid_size[2]-1] = 0
            self.grid_velocity_z[i, j, self.grid_size[2]] = 0

        for j, k in ti.ndrange(self.grid_size[1], self.grid_size[2]):
            self.grid_velocity_x[0, j, k] = 0
            self.grid_velocity_x[1, j, k] = 0
            self.grid_velocity_x[self.grid_size[0]-1, j, k] = 0
            self.grid_velocity_x[self.grid_size[0], j, k] = 0

        for i, k in ti.ndrange(self.grid_size[0], self.grid_size[2]):
            self.grid_velocity_y[i, 0, k] = 0
            self.grid_velocity_y[i, 1, k] = 0
            self.grid_velocity_y[i, self.grid_size[1]-1, k] = 0
            self.grid_velocity_y[i, self.grid_size[1], k] = 0

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
                        dpos = (ti.Vector([i-1, j-0.5, k-0.5]) - frac) * self.dx
                        self.grid_velocity_x[idx] += w * (cp @ dpos).x
                    self.grid_weight_x[idx] += w

        for i in ti.static(range(3)):
            for j in ti.static(range(4)):
                for k in ti.static(range(3)):
                    w = w_center[i].x * w_side[j].y * w_center[k].z
                    idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                    self.grid_velocity_y[idx] += vp.y * w
                    if self.mode == 'apic':
                        dpos = (ti.Vector([i-0.5, j-1, k-0.5]) - frac) * self.dx
                        self.grid_velocity_y[idx] += w * (cp @ dpos).y
                    self.grid_weight_y[idx] += w

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(4)):
                    w = w_center[i].x * w_center[j].y * w_side[k].z
                    idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                    self.grid_velocity_z[idx] += vp.z * w
                    if self.mode == 'apic':
                        dpos = (ti.Vector([i-0.5, j-0.5, k-1]) - frac) * self.dx
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
                        C_x += 4 * vtemp * dpos  / self.dx
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
                        C_y += 4 * vtemp * dpos / self.dx
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
                        C_z += 4 * vtemp * dpos / self.dx
                    wz += w

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
        if self.mode == 'apic':
            self.particles_affine_C[p] = ti.Matrix.rows([C_x/wx, C_y/wy, C_z/wz])

    @ti.kernel
    def compute_divergence(self):
        for i, j, k in self.divergence:
            if not self.is_solid(i, j, k):
                div = (self.grid_velocity_x[i + 1, j, k] - self.grid_velocity_x[i, j, k])
                div += (self.grid_velocity_y[i, j + 1, k] - self.grid_velocity_y[i, j, k])
                div += (self.grid_velocity_z[i, j, k + 1] - self.grid_velocity_z[i, j, k])

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

    @ti.kernel
    def compute_stress(self):
        for i, j, k in self.stress:
            if self.is_fluid(i, j, k):
                D = ti.Matrix.zero(ti.f32, 3, 3)
                # 2 * ux, uy + vx, uz + wx
                # vx + uy, 2 * vy, vz + wy
                # wx + uz, wy + vz, 2 * wz
                ux = self.grid_velocity_x[i + 1, j, k] - self.grid_velocity_x[i, j, k]
                uy = 0.25 * (self.grid_velocity_x[i, j + 1, k] + self.grid_velocity_x[i + 1, j + 1, k])\
                     - 0.25 * (self.grid_velocity_x[i, j - 1, k] + self.grid_velocity_x[i + 1, j - 1, k])
                uz = 0.25 * (self.grid_velocity_x[i, j, k + 1] + self.grid_velocity_x[i + 1, j, k + 1])\
                     - 0.25 * (self.grid_velocity_x[i, j, k - 1] + self.grid_velocity_x[i + 1, j, k - 1])
                vx = 0.25 * (self.grid_velocity_y[i + 1, j, k] + self.grid_velocity_y[i + 1, j + 1, k])\
                     - 0.25 * (self.grid_velocity_y[i - 1, j, k] + self.grid_velocity_y[i - 1, j + 1, k])
                vy = self.grid_velocity_y[i, j + 1, k] - self.grid_velocity_y[i, j, k]
                vz = 0.25 * (self.grid_velocity_y[i, j, k + 1] + self.grid_velocity_y[i, j + 1, k + 1]) \
                     - 0.25 * (self.grid_velocity_y[i, j, k - 1] + self.grid_velocity_y[i, j + 1, k - 1])
                wx = 0.25 * (self.grid_velocity_z[i + 1, j, k] + self.grid_velocity_z[i + 1, j, k + 1]) \
                     - 0.25 * (self.grid_velocity_z[i - 1, j, k] + self.grid_velocity_z[i - 1, j, k + 1])
                wy = 0.25 * (self.grid_velocity_x[i, j + 1, k] + self.grid_velocity_x[i, j + 1, k + 1]) \
                     - 0.25 * (self.grid_velocity_x[i, j - 1, k] + self.grid_velocity_x[i, j - 1, k + 1])
                wz = self.grid_velocity_z[i, j, k + 1] - self.grid_velocity_z[i, j, k]

                D[0, 0] = 2.0 * ux
                D[0, 1] = uy + vx
                D[0, 2] = uz + wx
                D[1, 0] = vx + uy
                D[1, 1] = 2.0 * vy
                D[1, 2] = vz + wy
                D[2, 0] = wx + uz
                D[2, 1] = wy + vz
                D[2, 2] = 2.0 * wz

                D = D / 2.0 / self.dx
                self.strain_rate[i, j, k] = D
                self.stress[i, j, k] = -self.rho * D * self.dx * self.dx / self.dt

    @ti.kernel
    def friction_condition(self):
        for i, j, k in self.stress:
            if self.is_fluid(i, j, k):
                D = self.strain_rate[i, j, k]
                D_F_norm = ti.sqrt(D[0,0]**2 + D[0,1]**2 + D[0,2]**2 + D[1,0]**2 + D[1,1]**2 + D[1,2]**2
                                 + D[2,0]**2 + D[2,1]**2 + D[2,2]**2)
                stress_f = ti.Matrix.zero(ti.f32, 3, 3)
                if D_F_norm > 1e-6:
                    stress_f = -self.mu * self.pressure[i, j, k] * D / (ti.sqrt(1.0/3.0) * D_F_norm)

                stress_m = self.stress[i, j, k].trace() / 3
                stress_bar = self.stress[i, j, k] - stress_m * ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                bar_F_norm = ti.sqrt(stress_bar[0,0]**2 + stress_bar[0,1]**2 + stress_bar[0,2]**2
                                   + stress_bar[1,0]**2 + stress_bar[1,1]**2 + stress_bar[1,2]**2
                                   + stress_bar[2,0]**2 + stress_bar[2,1]**2 + stress_bar[2,2]**2)
                if self.mu * self.pressure[i, j, k] <= bar_F_norm * ti.sqrt(3/2):
                    self.stress[i, j, k] = stress_f
                    self.sand_state[i, j, k] = 0 # flowing
                else:
                    self.sand_state[i, j, k] = 1 # rigidly moving

    def update_sand_rigid_group(self):
        states = self.sand_state.to_numpy()
        labels_out = cc3d.connected_components(states, connectivity=6)
        for label, group in cc3d.each(labels_out):
            self.group_label.from_numpy(group)
            self.update_sand_rigid_velocity()

    @ti.kernel
    def update_sand_rigid_velocity(self):
        # update velocity for each rigid group
        num = 0.0
        velocity = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        center_of_mass = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        L = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        I_00 = 0.0
        I_01 = 0.0
        I_02 = 0.0
        I_11 = 0.0
        I_12 = 0.0
        I_22 = 0.0

        for i, j, k in self.group_label:
            if self.group_label[i, j, k] != 0:
                ti.atomic_add(num, 1.0)
                pos = ti.Vector([i+0.5, j+0.5, k+0.5], dt=ti.f32)
                pos *= self.cell_extent
                ti.atomic_add(center_of_mass, pos)

        center_of_mass = center_of_mass / num

        for i, j, k in self.group_label:
            if self.group_label[i, j, k] != 0:
                v1 = (self.grid_velocity_x[i, j, k] + self.grid_velocity_x[i + 1, j, k]) / 2.0
                v2 = (self.grid_velocity_y[i, j, k] + self.grid_velocity_y[i, j + 1, k]) / 2.0
                v3 = (self.grid_velocity_z[i, j, k] + self.grid_velocity_z[i, j, k + 1]) / 2.0
                v = ti.Vector([v1, v2, v3], dt=ti.f32)
                ti.atomic_add(velocity, v)
                pos = ti.Vector([i + 0.5, j + 0.5, k + 0.5], dt=ti.f32)
                pos *= self.cell_extent
                r = pos - center_of_mass
                ti.atomic_add(L, r.cross(v))
                ti.atomic_add(I_00, r.x ** 2 + r.z ** 2)
                ti.atomic_add(I_01, -r.x * r.y)
                ti.atomic_add(I_02, -r.x * r.z)
                ti.atomic_add(I_11, r.x ** 2 + r.z ** 2)
                ti.atomic_add(I_12, -r.y * r.z)
                ti.atomic_add(I_22, r.x ** 2 + r.y ** 2)

        I = ti.Matrix.zero(ti.f32, 3, 3)
        I[0, 0] = I_00
        I[0, 1] = I_01
        I[0, 2] = I_02
        I[1, 0] = I_01
        I[1, 1] = I_11
        I[1, 2] = I_12
        I[2, 0] = I_02
        I[2, 1] = I_12
        I[2, 2] = I_22
        velocity = velocity / num
        w = I.inverse() @ L

        for i, j, k in self.group_label:
            if self.group_label[i, j, k] != 0:
                if num < 3 or ti.abs(w.x) > 1.0 or ti.abs(w.y) > 1.0 or ti.abs(w.z) > 1.0 :
                    rigid_velocity = velocity  # + (pos - center_of_mass).cross(w) #todo: add angular velocity
                    self.grid_velocity_x[i, j, k] = rigid_velocity.x
                    self.grid_velocity_x[i + 1, j, k] = rigid_velocity.x
                    self.grid_velocity_y[i, j, k] = rigid_velocity.y
                    self.grid_velocity_y[i, j + 1, k] = rigid_velocity.y
                    self.grid_velocity_z[i, j, k] = rigid_velocity.z
                    self.grid_velocity_z[i, j, k + 1] = rigid_velocity.z
                else:
                    pos = ti.Vector([float(i), float(j) + 0.5, float(k) + 0.5], dt=ti.f32)
                    pos *= self.cell_extent
                    rigid_velocity = velocity + (pos - center_of_mass).cross(w)
                    self.grid_velocity_x[i, j, k] = rigid_velocity.x

                    pos = ti.Vector([float(i) + 1.0, float(j) + 0.5, float(k) + 0.5], dt=ti.f32)
                    pos *= self.cell_extent
                    rigid_velocity = velocity + (pos - center_of_mass).cross(w)
                    self.grid_velocity_x[i + 1, j, k] = rigid_velocity.x

                    pos = ti.Vector([float(i) + 0.5, float(j), float(k) + 0.5], dt=ti.f32)
                    pos *= self.cell_extent
                    rigid_velocity = velocity + (pos - center_of_mass).cross(w)
                    self.grid_velocity_y[i, j, k] = rigid_velocity.y

                    pos = ti.Vector([float(i) + 0.5, float(j) + 1.0, float(k) + 0.5], dt=ti.f32)
                    pos *= self.cell_extent
                    rigid_velocity = velocity + (pos - center_of_mass).cross(w)
                    self.grid_velocity_y[i, j + 1, k] = rigid_velocity.y

                    pos = ti.Vector([float(i) + 0.5, float(j) + 0.5, float(k)], dt=ti.f32)
                    pos *= self.cell_extent
                    rigid_velocity = velocity + (pos - center_of_mass).cross(w)
                    self.grid_velocity_z[i, j, k] = rigid_velocity.z

                    pos = ti.Vector([float(i) + 0.5, float(j) + 0.5, float(k) + 1.0], dt=ti.f32)
                    pos *= self.cell_extent
                    rigid_velocity = velocity + (pos - center_of_mass).cross(w)
                    self.grid_velocity_z[i, j, k + 1] = rigid_velocity.z

    @ti.kernel
    def update_sand_flowing(self):
        scale = self.dt / (self.rho * self.dx)
        for i, j, k in self.sand_state:
            dsx = self.stress[i, j, k] - self.stress[i - 1, j, k]
            dsy = 0.25 * (self.stress[i, j + 1, k] + self.stress[i - 1, j + 1, k]) \
                  - 0.25 * (self.stress[i, j - 1, k] + self.stress[i - 1, j - 1, k])
            dsz = 0.25 * (self.stress[i, j, k + 1] + self.stress[i - 1, j, k + 1]) \
                  - 0.25 * (self.stress[i, j, k - 1] + self.stress[i - 1, j, k - 1])
            if self.is_fluid(i - 1, j, k) and self.sand_state[i - 1, j, k] == 0 \
                or self.is_fluid(i, j, k) and self.sand_state[i, j, k] == 0:
                if self.is_solid(i - 1, j, k) or self.is_solid(i, j, k):
                    self.grid_velocity_x[i, j, k] = 0
                else:
                    self.grid_velocity_x[i, j, k] -= scale * (dsx[0, 0] + dsy[0, 1] + dsz[0, 2])

            dsx = 0.25 * (self.stress[i + 1, j, k] + self.stress[i + 1, j - 1, k]) \
                  - 0.25 * (self.stress[i - 1, j, k] + self.stress[i - 1, j - 1, k])
            dsy = self.stress[i, j, k] - self.stress[i, j - 1, k]
            dsz = 0.25 * (self.stress[i, j, k + 1] + self.stress[i, j - 1, k + 1]) \
                  - 0.25 * (self.stress[i, j, k - 1] + self.stress[i, j - 1, k - 1])
            if self.is_fluid(i, j - 1, k) and self.sand_state[i, j - 1, k] == 0 \
                or self.is_fluid(i, j, k) and self.sand_state[i, j, k] == 0:
                if self.is_solid(i, j - 1, k) or self.is_solid(i, j, k):
                    self.grid_velocity_y[i, j, k] = 0
                else:
                    self.grid_velocity_y[i, j, k] -= scale * (dsx[1, 0] + dsy[1, 1] + dsz[1, 2])

            dsx = 0.25 * (self.stress[i + 1, j, k] + self.stress[i + 1, j, k - 1]) \
                  - 0.25 * (self.stress[i - 1, j, k] + self.stress[i - 1, j, k - 1])
            dsy = 0.25 * (self.stress[i, j + 1, k] + self.stress[i, j + 1, k - 1]) \
                  - 0.25 * (self.stress[i, j - 1, k] + self.stress[i, j - 1, k - 1])
            dsz = self.stress[i, j, k] - self.stress[i, j, k - 1]
            if self.is_fluid(i, j, k - 1) and self.sand_state[i, j, k - 1] == 0 \
                or self.is_fluid(i, j, k) and self.sand_state[i, j, k] == 0:
                if self.is_solid(i, j, k - 1) or self.is_solid(i, j, k):
                    self.grid_velocity_z[i, j, k] = 0
                else:
                    self.grid_velocity_z[i, j, k] -= scale * (dsx[2, 0] + dsy[2, 1] + dsz[2, 2])



