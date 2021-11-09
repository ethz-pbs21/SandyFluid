import taichi as ti
import numpy as np
import math

ti.init(default_fp=ti.f32, arch=ti.opengl, kernel_profiler=True)

N = 256
nx = N
ny = N
res_x = 512
res_y = res_x * ny // nx

use_flip = True
save_results = True

gravity = -9.8
flip_viscosity = 0.01

SOLID = 2
AIR = 1
FLUID = 0

FLIP99 = 0
PIC = 1
APIC = 2

@ti.func
def clamp(x, a, b):
    return max(a, min(b, x))


@ti.data_oriented
class ColorMap:
    def __init__(self, h, wl, wr, c):
        self.h = h
        self.wl = wl
        self.wr = wr
        self.c = c

    @ti.func
    def clamp(self, x):
        return max(0.0, min(1.0, x))

    @ti.func
    def map(self, x):
        w = 0.0
        if x < self.c:
            w = self.wl
        else:
            w = self.wr
        return self.clamp((w-abs(self.clamp(x)-self.c))/w*self.h)


jetR = ColorMap(1.5, .37, .37, .75)
jetG = ColorMap(1.5, .37, .37, .5)
jetB = ColorMap(1.5, .37, .37, .25)

bwrR = ColorMap(1.0, .25, 1, .5)
bwrG = ColorMap(1.0, .5, .5, .5)
bwrB = ColorMap(1.0, 1, .25, .5)


@ti.func
def color_map(c):
    # return ti.Vector([jetR.map(c),
    #                   jetG.map(c),
    #                   jetB.map(c)])
    return ti.Vector([bwrR.map(c),
                      bwrG.map(c),
                      bwrB.map(c)])

# data : field to sample from
# u : x coord of sample location [0, nx]
# v : y coord of sample location [0, ny]
# ox : x coord of data[0, 0], e.g. 0.5 for cell-centered data
# oy : y coord of data[0, 0], e.g. 0.5 for cell-centered data
# nx : x resolution of data
# ny : y resolution of data
@ti.func
def sample(data, u, v, ox, oy, nx, ny):
    s, t = u - ox, v - oy
    i, j = clamp(int(s), 0, nx - 1), clamp(int(t), 0, ny - 1)
    ip, jp = clamp(i + 1, 0, nx - 1), clamp(j + 1, 0, ny - 1)
    s, t = clamp(s - i, 0.0, 1.0), clamp(t - j, 0.0, 1.0)
    return \
        (data[i, j] * (1 - s) + data[ip, j] * s) * (1 - t) + \
        (data[i, jp] * (1 - s) + data[ip, jp] * s) * t


# data : value field to splat to
# weights : weight field to splat to
# f: value of the sample to splat
# u : x coord of sample location [0, nx]
# v : y coord of sample location [0, ny]
# ox : x coord of data[0, 0], e.g. 0.5 for cell-centered data
# oy : y coord of data[0, 0], e.g. 0.5 for cell-centered data
# nx : x resolution of data
# ny : y resolution of data
@ti.func
def splat(data, weights, f, u, v, ox, oy, nx, ny, cp):
    s, t = u - ox, v - oy
    i, j = clamp(int(s), 0, nx - 1), clamp(int(t), 0, ny - 1)
    ip, jp = clamp(i + 1, 0, nx - 1), clamp(j + 1, 0, ny - 1)
    s1, t1 = clamp(s - i, 0.0, 1.0), clamp(t - j, 0.0, 1.0)
    dpos0 = ti.Vector([s - i, t - j])
    dpos1 = ti.Vector([s - ip, t - j])
    dpos2 = ti.Vector([s - i, t - jp])
    dpos3 = ti.Vector([s - ip, t - jp])
    data[i, j] += (f + cp.dot(dpos0)) * (1 - s1) * (1 - t1)
    data[ip, j] += (f + cp.dot(dpos1)) * (s1) * (1 - t1)
    data[i, jp] += (f + cp.dot(dpos2)) * (1 - s1) * (t1)
    data[ip, jp] += (f + cp.dot(dpos3)) * (s1) * (t1)
    weights[i, j] += (1 - s1) * (1 - t1)
    weights[ip, j] += (s1) * (1 - t1)
    weights[i, jp] += (1 - s1) * (t1)
    weights[ip, jp] += (s1) * (t1)


@ti.data_oriented
class Texture:
    def __init__(self, data, ox, oy, nx, ny):
        self.data = data
        self.ox = ox
        self.oy = oy
        self.nx = nx
        self.ny = ny

    @ti.func
    def sample(self, u, v):
        return sample(self.data, u, v, self.ox, self.oy, self.nx, self.ny)


@ti.data_oriented
class MultigridPCGPoissonSolver:
    def __init__(self, marker, nx, ny):
        shape = (nx, ny)
        self.nx, self.ny = shape
        print(f'nx, ny = {nx}, {ny}')

        self.dim = 2
        self.max_iters = 300
        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.use_multigrid = True

        def _res(l): return (nx // (2**l), ny // (2**l))

        self.r = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # M^-1 r
        self.d = [ti.field(dtype=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # temp
        self.f = [marker] + [ti.field(dtype=ti.i32, shape=_res(_))
                             for _ in range(self.n_mg_levels - 1)]  # marker
        self.L = [ti.Vector(6, dt=ti.f32, shape=_res(_))
                  for _ in range(self.n_mg_levels)]  # -L operator

        self.x = ti.field(dtype=ti.f32, shape=shape)  # solution
        self.p = ti.field(dtype=ti.f32, shape=shape)  # conjugate gradient
        self.Ap = ti.field(dtype=ti.f32, shape=shape)  # matrix-vector product
        self.alpha = ti.field(dtype=ti.f32, shape=())  # step size
        self.beta = ti.field(dtype=ti.f32, shape=())  # step size
        self.sum = ti.field(dtype=ti.f32, shape=())  # storage for reductions

        for _ in range(self.n_mg_levels):
            print(f'r[{_}].shape = {self.r[_].shape}')
        for _ in range(self.n_mg_levels):
            print(f'L[{_}].shape = {self.L[_].shape}')

    @ti.func
    def is_fluid(self, f, i, j, nx, ny):
        return i >= 0 and i < nx and j >= 0 and j < ny and FLUID == f[i, j]

    @ti.func
    def is_solid(self, f, i, j, nx, ny):
        return i < 0 or i >= nx or j < 0 or j >= ny or SOLID == f[i, j]

    @ti.func
    def is_air(self, f, i, j, nx, ny):
        return i >= 0 and i < nx and j >= 0 and j < ny and AIR == f[i, j]

    @ti.func
    def neighbor_sum(self, L, x, f, i, j, nx, ny):
        ret = x[(i - 1 + nx) % nx, j] * L[i, j][2]
        ret += x[(i + 1 + nx) % nx, j] * L[i, j][3]
        ret += x[i, (j - 1 + ny) % ny] * L[i, j][4]
        ret += x[i, (j + 1 + ny) % ny] * L[i, j][5]
        return ret

    # -L matrix : 0-diagonal, 1-diagonal inverse, 2...-off diagonals
    @ti.kernel
    def init_L(self, l: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in self.L[l]:
            if FLUID == self.f[l][i, j]:
                s = 4.0
                s -= float(self.is_solid(self.f[l], i - 1, j, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i + 1, j, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i, j - 1, _nx, _ny))
                s -= float(self.is_solid(self.f[l], i, j + 1, _nx, _ny))
                self.L[l][i, j][0] = s
                self.L[l][i, j][1] = 1.0 / s
            self.L[l][i, j][2] = float(
                self.is_fluid(self.f[l], i - 1, j, _nx, _ny))
            self.L[l][i, j][3] = float(
                self.is_fluid(self.f[l], i + 1, j, _nx, _ny))
            self.L[l][i, j][4] = float(
                self.is_fluid(self.f[l], i, j - 1, _nx, _ny))
            self.L[l][i, j][5] = float(
                self.is_fluid(self.f[l], i, j + 1, _nx, _ny))

    def solve(self, x, rhs):
        tol = 1e-12

        self.r[0].copy_from(rhs)
        self.x.fill(0.0)

        self.Ap.fill(0.0)
        self.p.fill(0.0)

        for l in range(1, self.n_mg_levels):
            self.downsample_f(self.f[l - 1], self.f[l],
                              self.nx // (2**l), self.ny // (2**l))
        for l in range(self.n_mg_levels):
            self.L[l].fill(0.0)
            self.init_L(l)

        self.sum[None] = 0.0
        self.reduction(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        print(f"init rtr = {initial_rTr}")

        if initial_rTr < tol:
            print(f"converged: init rtr = {initial_rTr}")
        else:
            # r = b - Ax = b    since x = 0
            # p = r = r + 0 p
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            self.update_p()

            self.sum[None] = 0.0
            self.reduction(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iter = 0
            for i in range(self.max_iters):
                # alpha = rTr / pTAp
                self.apply_L(0, self.p, self.Ap)

                self.sum[None] = 0.0
                self.reduction(self.p, self.Ap)
                pAp = self.sum[None]

                self.alpha[None] = old_zTr / pAp

                # x = x + alpha p
                # r = r - alpha Ap
                self.update_x_and_r()

                # check for convergence
                self.sum[None] = 0.0
                self.reduction(self.r[0], self.r[0])
                rTr = self.sum[None]
                if rTr < initial_rTr * tol:
                    break

                # z = M^-1 r
                if self.use_multigrid:
                    self.apply_preconditioner()
                else:
                    self.z[0].copy_from(self.r[0])

                # beta = new_rTr / old_rTr
                self.sum[None] = 0.0
                self.reduction(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                self.beta[None] = new_zTr / old_zTr

                # p = z + beta p
                self.update_p()
                old_zTr = new_zTr

                iter = i
            print(f'converged to {rTr} in {iter} iters')

        x.copy_from(self.x)

    @ti.kernel
    def apply_L(self, l: ti.template(), x: ti.template(), Ax: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in Ax:
            if FLUID == self.f[l][i, j]:
                r = x[i, j] * self.L[l][i, j][0]
                r -= self.neighbor_sum(self.L[l], x,
                                       self.f[l], i, j, _nx, _ny)
                Ax[i, j] = r

    @ti.kernel
    def reduction(self, p: ti.template(), q: ti.template()):
        for I in ti.grouped(p):
            if FLUID == self.f[0][I]:
                self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x_and_r(self):
        a = float(self.alpha[None])
        for I in ti.grouped(self.p):
            if FLUID == self.f[0][I]:
                self.x[I] += a * self.p[I]
                self.r[0][I] -= a * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            if FLUID == self.f[0][I]:
                self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    # ------------------ multigrid ---------------
    @ti.kernel
    def downsample_f(self, f_fine: ti.template(), f_coarse: ti.template(),
                     nx: ti.template(), ny: ti.template()):
        for i, j in f_coarse:
            i2 = i * 2
            j2 = j * 2

            if AIR == f_fine[i2, j2] or AIR == f_fine[i2 + 1, j2] or \
               AIR == f_fine[i2, j2 + 1] or AIR == f_fine[i2 + 1, j2 + 1]:
                f_coarse[i, j] = AIR
            else:
                if FLUID == f_fine[i2, j2] or FLUID == f_fine[i2 + 1, j2] or \
                   FLUID == f_fine[i2 + 1, j2] or FLUID == f_fine[i2 + 1, j2 + 1]:
                    f_coarse[i, j] = FLUID
                else:
                    f_coarse[i, j] = SOLID

    @ti.kernel
    def restrict(self, l: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in self.r[l]:
            if FLUID == self.f[l][i, j]:
                Az = self.L[l][i, j][0] * self.z[l][i, j]
                Az -= self.neighbor_sum(self.L[l],
                                        self.z[l], self.f[l], i, j, _nx, _ny)
                res = self.r[l][i, j] - Az
                self.r[l + 1][i // 2, j // 2] += res

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] += self.z[l + 1][I // 2]

    # Gause-Seidel
    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        _nx, _ny = self.nx // (2**l), self.ny // (2**l)
        for i, j in self.r[l]:
            if FLUID == self.f[l][i, j] and (i + j) & 1 == phase:
                self.z[l][i, j] = (self.r[l][i, j]
                                   + self.neighbor_sum(self.L[l], self.z[l], self.f[l], i, j, _nx, _ny)
                                   ) * self.L[l][i, j][1]

    def apply_preconditioner(self):

        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.d[l].fill(0.0)
            self.restrict(l)

        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)
        for i in range(self.bottom_smoothing // 2):
            self.smooth(self.n_mg_levels - 1, 1)
            self.smooth(self.n_mg_levels - 1, 0)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)


@ti.data_oriented
class Swapper:
    def __init__(self, data, data_temp, data_tex, data_temp_tex):
        self.data = data
        self.data_temp = data_temp
        self.data_tex = data_tex
        self.data_temp_tex = data_temp_tex

    def swap(self):
        self.data, self.data_temp = self.data_temp, self.data
        self.data_tex, self.data_temp_tex = self.data_temp_tex, self.data_tex


class FlipSolver:
    pass


pressure = ti.field(dtype=ti.f32, shape=(nx, ny))
pressure_tex = Texture(pressure, 0.5, 0.5, nx, ny)

divergence = ti.field(dtype=ti.f32, shape=(nx, ny))
divergence_tex = Texture(divergence, 0.5, 0.5, nx, ny)

marker = ti.field(dtype=ti.i32, shape=(nx, ny))

ux = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
uy = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
ux_temp = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
uy_temp = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
ux_saved = ti.field(dtype=ti.f32, shape=(nx + 1, ny))
uy_saved = ti.field(dtype=ti.f32, shape=(nx, ny + 1))
# ux_tex = Texture(ux, 0.0, 0.5, nx + 1, ny)
# uy_tex = Texture(uy, 0.5, 0.0, nx, ny + 1)
# ux_temp_tex = Texture(ux_temp, 0.0, 0.5, nx + 1, ny)
# uy_temp_tex = Texture(uy_temp, 0.5, 0.0, nx, ny + 1)

cp_x = ti.Vector(2, dt=ti.f32, shape=(nx * 2, ny * 2))
cp_y = ti.Vector(2, dt=ti.f32, shape=(nx * 2, ny * 2))
# ux_swap = Swapper(ux, ux_temp, ux_tex, ux_temp_tex)
# uy_swap = Swapper(uy, uy_temp, uy_tex, uy_temp_tex)

px = ti.Vector(2, dt=ti.f32, shape=(nx * 2, ny * 2))
pv = ti.Vector(2, dt=ti.f32, shape=(nx * 2, ny * 2))
pf = ti.field(dtype=ti.i32, shape=(nx * 2, ny * 2))

valid = ti.field(dtype=ti.i32, shape=(nx + 1, ny + 1))
valid_temp = ti.field(dtype=ti.i32, shape=(nx + 1, ny + 1))

color_buffer = ti.Vector(3, dt=ti.f32, shape=(res_x, res_y))


ps = MultigridPCGPoissonSolver(marker, nx, ny)


@ti.func
def is_fluid(i, j):
    return i >= 0 and i < nx and j >= 0 and j < ny and FLUID == marker[i, j]


@ti.func
def is_solid(i, j):
    return i < 0 or i >= nx or j < 0 or j >= ny or SOLID == marker[i, j]


@ti.func
def is_air(i, j):
    return i >= 0 and i < nx and j >= 0 and j < ny and AIR == marker[i, j]


@ti.func
def vel_interp(pos, ux, uy):
    _ux = sample(ux, pos.x, pos.y, 0.0, 0.5, nx + 1, ny)
    _uy = sample(uy, pos.x, pos.y, 0.5, 0.0, nx, ny + 1)
    return ti.Vector([_ux, _uy])


# move particles with the divergence-free grid velocity
@ti.kernel
def advect_markers(dt: ti.f32):
    for i, j in px:
        if 1 == pf[i, j]:
            midpos = px[i, j] + vel_interp(px[i, j], ux, uy) * (dt * 0.5)
            px[i, j] += vel_interp(midpos, ux, uy) * dt


@ti.kernel
def apply_markers():
    for i, j in marker:
        if SOLID != marker[i, j]:
            marker[i, j] = AIR

    for m, n in px:
        if 1 == pf[m, n]:
            i = clamp(int(px[m, n].x), 0, nx-1)
            j = clamp(int(px[m, n].y), 0, ny-1)
            if (SOLID != marker[i, j]):
                marker[i, j] = FLUID


@ti.kernel
def add_force(dt: ti.f32):
    for i, j in uy:
        uy[i, j] += gravity * dt


@ti.kernel
def apply_bc():
    for i, j in marker:
        if SOLID == marker[i, j]:
            ux[i, j] = 0.0
            ux[i + 1, j] = 0.0
            uy[i, j] = 0.0
            uy[i, j + 1] = 0.0

# from the Helmholtz decomposition
# solve -L p = -div then apply -grad p
@ti.kernel
def calc_divergence(ux: ti.template(), uy: ti.template(), div: ti.template(), marker: ti.template()):
    for i, j in div:
        if FLUID == marker[i, j]:
            _dx = ux[i, j] - ux[i + 1, j]
            _dy = uy[i, j] - uy[i, j + 1]
            div[i, j] = _dx + _dy


def solve_pressure():
    divergence.fill(0.0)
    calc_divergence(ux, uy, divergence, marker)
    pressure.fill(0.0)
    ps.solve(pressure, divergence)


@ti.kernel
def apply_pressure():
    for i, j in ux:
        if is_fluid(i - 1, j) or is_fluid(i, j):
            ux[i, j] += pressure[i - 1, j] - pressure[i, j]

    for i, j in uy:
        if is_fluid(i, j - 1) or is_fluid(i, j):
            uy[i, j] += pressure[i, j - 1] - pressure[i, j]


@ti.kernel
def advect(dst: ti.template(), src: ti.template(), dt: ti.f32,
           ox: ti.f32, oy: ti.f32, nx: ti.i32, ny: ti.i32):
    for i, j in dst:
        pos = ti.Vector([i + ox, j + oy])
        midpos = pos - vel_interp(pos, ux, uy) * (dt * 0.5)
        p0 = pos - vel_interp(midpos, ux, uy) * dt
        dst[i, j] = sample(src, p0.x, p0.y, ox, oy, nx, ny)


@ti.kernel
def mark_valid_ux():
    for i, j in ux:
        # NOTE that the the air-liquid interface is valid
        if is_fluid(i - 1, j) or is_fluid(i, j):
            valid[i, j] = 1
        else:
            valid[i, j] = 0


@ti.kernel
def mark_valid_uy():
    for i, j in uy:
        # NOTE that the the air-liquid interface is valid
        if is_fluid(i, j - 1) or is_fluid(i, j):
            valid[i, j] = 1
        else:
            valid[i, j] = 0


@ti.kernel
def diffuse_quantity(dst: ti.template(), src: ti.template(),
                     valid_dst: ti.template(), valid: ti.template()):
    for i, j in dst:
        if 0 == valid[i, j]:
            sum = 0.0
            count = 0
            for m, n in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                if 1 == valid[i + m, j + n]:
                    sum += src[i + m, j + n]
                    count += 1
            if count > 0:
                dst[i, j] = sum / float(count)
                valid_dst[i, j] = 1


def extrap_velocity():
    mark_valid_ux()
    for i in range(10):
        ux_temp.copy_from(ux)
        valid_temp.copy_from(valid)
        diffuse_quantity(ux, ux_temp, valid, valid_temp)

    mark_valid_uy()
    for i in range(10):
        uy_temp.copy_from(uy)
        valid_temp.copy_from(valid)
        diffuse_quantity(uy, uy_temp, valid, valid_temp)


# data : field to sample from
# u : x coord of sample location [0, nx]
# v : y coord of sample location [0, ny]
# ox : x coord of data[0, 0], e.g. 0.5 for cell-centered data
# oy : y coord of data[0, 0], e.g. 0.5 for cell-centered data
# nx : x resolution of data
# ny : y resolution of data

@ti.func
def apic_c(stagger, xp, grid_v):        
    base = (xp - stagger).cast(ti.i32)
    new_c = ti.Vector([0.0, 0.0])
    
    for i, j in ti.static(ti.ndrange(2, 2)):

        offset = ti.Vector([i, j])
        fx = xp - (base + offset + stagger)
        dpos = fx
        weight = (1 - abs(fx.x)) * (1 - abs(fx.y))
        new_c += 4 * weight * dpos * grid_v[base + offset]

    return new_c

@ti.kernel
def update_from_grid():
    stagger_u = ti.Vector([0, 0.5])
    stagger_v = ti.Vector([0.5, 0])
    for m, n in px:
        if 1 == pf[m, n]:
            gvel = vel_interp(px[m, n], ux, uy)
            pv[m, n] = gvel
            cp_x[m, n] = apic_c(stagger_u, px[m, n], ux)
            cp_y[m, n] = apic_c(stagger_v, px[m, n], uy)




@ti.kernel
def transfer_to_grid(weights_ux: ti.template(), weights_uy: ti.template()):
    for m, n in pv:
        if 1 == pf[m, n]:
            x, y = px[m, n].x, px[m, n].y
            u, v = pv[m, n].x, pv[m, n].y
            splat(ux, weights_ux, u, x, y, 0.0, 0.5, nx + 1, ny, cp_x[m, n])
            splat(uy, weights_uy, v, x, y, 0.5, 0.0, nx, ny + 1, cp_y[m, n])

    for i, j in weights_ux:
        if weights_ux[i, j] > 0.0:
            ux[i, j] /= weights_ux[i, j]

    for i, j in weights_uy:
        if weights_uy[i, j] > 0.0:
            uy[i, j] /= weights_uy[i, j]


def save_velocities():
    ux_saved.copy_from(ux)
    uy_saved.copy_from(uy)


def substep(dt):
    add_force(dt)
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


idx = 0


def step():
    global idx
    print(f'frame {idx}')
    idx = idx + 1
    # substep(1.0 / 20)
    # substep(1.0 / 20)
    substep(1.0 / 80)
    substep(1.0 / 80)
    substep(1.0 / 80)
    substep(1.0 / 80)


@ti.kernel
def fill_pressure():
    for i, j in color_buffer:
        x = (i + 0.5) * nx / res_x
        y = (j + 0.5) * ny / res_y
        _pressure = pressure_tex.sample(x, y)
        color_buffer[i, j] = color_map(_pressure * 0.1)


@ti.kernel
def fill_particles():
    for i, j in px:
        x = int(px[i, j].x / nx * res_x)
        y = int(px[i, j].y / ny * res_y)

        color_buffer[x, y] = ti.Vector([1.0, 1.0, 1.0])


@ti.kernel
def fill_marker():
    for i, j in color_buffer:
        x = int((i + 0.5) * nx / res_x)
        y = int((j + 0.5) * ny / res_y)
        m = marker[x, y]

        color_buffer[i, j] = color_map(m * 0.5)


@ti.kernel
def init_still_water_test():
    for i, j in pressure:
        pressure[i, j] = 0
        divergence[i, j] = 0

        if (j > ny // 2):
            marker[i, j] = AIR
        else:
            marker[i, j] = FLUID

        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            marker[i, j] = SOLID


@ti.kernel
def init_still_water_ceiling():
    for i, j in pressure:
        pressure[i, j] = 0
        divergence[i, j] = 0

        if (j < ny // 2):
            marker[i, j] = AIR
        else:
            marker[i, j] = FLUID

        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            marker[i, j] = SOLID


@ti.kernel
def init_dambreak():
    for i, j in pressure:
        pressure[i, j] = 0
        divergence[i, j] = 0

        if (j > ny // 2) or ( nx //4 * 3 < i or i < nx // 4):
            marker[i, j] = AIR
        else:
            marker[i, j] = FLUID

        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            marker[i, j] = SOLID


@ti.kernel
def init_waterfall():
    for i, j in pressure:
        pressure[i, j] = 0
        divergence[i, j] = 0

        if (j > ny * 2 // 3 and j < ny - 2):
            marker[i, j] = FLUID
        else:
            marker[i, j] = AIR

        if (i == 0 or i == nx-1 or j == 0 or j == ny-1) \
                or (j == ny * 2 // 3 and i > nx // 3) \
                or (j == ny // 3 and i < nx * 2 // 3):
            marker[i, j] = SOLID


def init_fields():
    init_dambreak()


@ti.kernel
def init_particles():
    for m, n in px:
        i, j = m // 2, n // 2
        px[m, n] = [0.0, 0.0]
        if FLUID == marker[i, j]:
            pf[m, n] = 1

            x = i + ((m % 2) + 0.5) / 2.0
            y = j + ((n % 2) + 0.5) / 2.0

            px[m, n] = [x, y]


def initialize():
    ux.fill(0.0)
    uy.fill(0.0)
    cp_x.fill(ti.Vector([0.0, 0.0]))
    cp_y.fill(ti.Vector([0.0, 0.0]))
    pf.fill(0)
    px.fill(ti.Vector([0.0, 0.0]))
    pv.fill(ti.Vector([0.0, 0.0]))

    init_fields()
    init_particles()


@ti.data_oriented
class Viewer:
    def __init__(self, dump):
        self.display_mode = 0
        self.is_active = True
        self.dump = dump
        self.frame = 0

        if self.dump:
            result_dir = "./results"
            self.video_manager = ti.VideoManager(
                output_dir=result_dir, framerate=24, automatic_build=False)

    def toggle(self):
        self.display_mode = (self.display_mode + 1) % 3

    def active(self):
        return self.is_active

    def draw(self, gui):
        # gui.clear(0x0)
        gui.circles(np.reshape(px.to_numpy(), (nx * ny * 4, 2)) *
                     np.array([[1.0 / nx, 1.0 / nx]]), radius=1, color=0x0FFFFF)
        gui.show(f'{self.frame:06d}.png')

        self.frame = self.frame + 1


def main():
    initialize()

    viewer = Viewer(save_results)
    cnt = 0
    gui = ti.GUI("flip", res=(res_x, res_y))
    while viewer.active():
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE or e.key == 'q':
                exit(0)

        step()
        viewer.draw(gui)

        gui.show()
        #gui.cook_image
        if cnt >=1000:
            break

    ti.kernel_profiler_print()


# if __name__ == 'main':
main()
