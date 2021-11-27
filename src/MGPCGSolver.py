import taichi as ti

#define cell type
FLUID = 0
AIR = 1
SOLID = 2


@ti.data_oriented
class MGPCGSolver:
    def __init__(self,
                 grid_size,
                 u,
                 v,
                 w,
                 cell_type,
                 multigrid_level=1,
                 pre_and_post_smoothing=2,
                 bottom_smoothing=10):
        self.grid_size = grid_size
        self.u = u
        self.v = v
        self.w = w
        self.cell_type = cell_type
        self.multigrid_level = multigrid_level
        self.pre_and_post_smoothing = pre_and_post_smoothing
        self.bottom_smoothing = bottom_smoothing

        # rhs of linear system
        self.b = ti.field(dtype=ti.f32,
                          shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2]))

        def grid_shape(l):
            return (self.grid_size[0] // 2**l, self.grid_size[1] // 2**l,
                    self.grid_size[2] // 2**l)

        # lhs of linear system and its corresponding form in coarse grids
        self.Adiag = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]
        self.Ax = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]
        self.Ay = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]
        self.Az = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]

        # grid type
        self.grid_type = [
            ti.field(dtype=ti.i32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]

        # pcg var
        self.r = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]
        self.z = [
            ti.field(dtype=ti.f32, shape=grid_shape(l))
            for l in range(self.multigrid_level)
        ]

        self.p = ti.field(dtype=ti.f32,
                          shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2]))
        self.s = ti.field(dtype=ti.f32,
                          shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2]))
        self.As = ti.field(dtype=ti.f32,
                           shape=(self.grid_size[0], self.grid_size[1], self.grid_size[2]))
        self.sum = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.beta = ti.field(dtype=ti.f32, shape=())

    def system_init(self, scale_A, scale_b):
        self.b.fill(0.0)

        for l in range(self.multigrid_level):
            self.Adiag[l].fill(0.0)
            self.Ax[l].fill(0.0)
            self.Ay[l].fill(0.0)
            self.Az[l].fill(0.0)

        self.system_init_kernel(scale_A, scale_b)
        self.grid_type[0].copy_from(self.cell_type)

        for l in range(1, self.multigrid_level):
            self.gridtype_init(l)
            self.preconditioner_init(scale_A, l)

    @ti.kernel
    def system_init_kernel(self, scale_A: ti.f32, scale_b: ti.f32):
        #define right hand side of linear system
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                self.b[i, j, k] = -1 * scale_b * (
                    self.u[i + 1, j, k] - self.u[i, j, k] + self.v[i, j + 1, k]
                    - self.v[i, j, k] + self.w[i, j, k + 1] - self.w[i, j, k])

        #modify right hand side of linear system to account for solid velocities
        #currently hard code solid velocities to zero
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                if self.cell_type[i - 1, j, k] == SOLID:
                    self.b[i, j, k] -= scale_b * (self.u[i, j, k] - 0)
                if self.cell_type[i + 1, j, k] == SOLID:
                    self.b[i, j, k] += scale_b * (self.u[i + 1, j, k] - 0)

                if self.cell_type[i, j - 1, k] == SOLID:
                    self.b[i, j, k] -= scale_b * (self.v[i, j, k] - 0)
                if self.cell_type[i, j + 1, k] == SOLID:
                    self.b[i, j, k] += scale_b * (self.v[i, j + 1, k] - 0)

                if self.cell_type[i, j, k - 1] == SOLID:
                    self.b[i, j, k] -= scale_b * (self.w[i, j, k] - 0)
                if self.cell_type[i, j, k + 1] == SOLID:
                    self.b[i, j, k] += scale_b * (self.v[i, j, k + 1] - 0)

        # define left handside of linear system
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                if self.cell_type[i - 1, j, k] == FLUID:
                    self.Adiag[0][i, j, k] += scale_A
                if self.cell_type[i + 1, j, k] == FLUID:
                    self.Adiag[0][i, j, k] += scale_A
                    self.Ax[0][i, j, k] = -scale_A
                elif self.cell_type[i + 1, j, k] == AIR:
                    self.Adiag[0][i, j, k] += scale_A

                if self.cell_type[i, j - 1, k] == FLUID:
                    self.Adiag[0][i, j, k] += scale_A
                if self.cell_type[i, j + 1, k] == FLUID:
                    self.Adiag[0][i, j, k] += scale_A
                    self.Ay[0][i, j, k] = -scale_A
                elif self.cell_type[i, j + 1, k] == AIR:
                    self.Adiag[0][i, j, k] += scale_A

                if self.cell_type[i, j, k - 1] == FLUID:
                    self.Adiag[0][i, j, k] += scale_A
                if self.cell_type[i, j, k + 1] == FLUID:
                    self.Adiag[0][i, j, k] += scale_A
                    self.Az[0][i, j, k] = -scale_A
                elif self.cell_type[i, j, k + 1] == AIR:
                    self.Adiag[0][i, j, k] += scale_A

    @ti.kernel
    def gridtype_init(self, l: ti.template()):
        for i, j, k in self.grid_type[l]:
            i2 = i * 2
            j2 = j * 2
            k2 = k * 2

            if self.grid_type[l - 1][i2, j2, k2] == AIR or self.grid_type[
                    l - 1][i2, j2 + 1, k2] == AIR or self.grid_type[l - 1][
                        i2 + 1, j2, k2] == AIR or self.grid_type[l - 1][
                            i2 + 1, j2 + 1,
                            k2] == AIR or self.grid_type[l - 1][
                                i2, j2,
                                k2 + 1] == AIR or self.grid_type[l - 1][
                                    i2, j2 + 1,
                                    k2 + 1] == AIR or self.grid_type[l - 1][
                                        i2 + 1, j2, k2 +
                                        1] == AIR or self.grid_type[l - 1][
                                            i2 + 1, j2 + 1, k2 + 1] == AIR:
                self.grid_type[l][i, j, k] = AIR
            else:
                if self.grid_type[l - 1][i2, j2, k2] == FLUID or self.grid_type[
                        l -
                        1][i2, j2 + 1, k2] == FLUID or self.grid_type[l - 1][
                            i2 + 1, j2, k2] == FLUID or self.grid_type[l - 1][
                                i2 + 1, j2 + 1,
                                k2] == FLUID or self.grid_type[l - 1][
                                    i2,
                                    j2,
                                    k2 + 1] == FLUID or self.grid_type[l - 1][
                                        i2,
                                        j2 + 1,
                                        k2 +
                                        1] == FLUID or self.grid_type[l - 1][
                                            i2 + 1,
                                            j2,
                                            k2 + 1] == FLUID or self.grid_type[
                                                l - 1][i2 + 1, j2 + 1,
                                                       k2 + 1] == FLUID:
                    self.grid_type[l][i, j, k] = FLUID
                else:
                    self.grid_type[l][i, j, k] = SOLID

    @ti.kernel
    def preconditioner_init(self, scale: ti.f32, l: ti.template()):
        scale = scale / (2**l * 2**l)

        for i, j, k in self.grid_type[l]:
            if self.grid_type[l][i, j, k] == FLUID:
                if self.grid_type[l][i - 1, j, k] == FLUID:
                    self.Adiag[l][i, j, k] += scale
                if self.grid_type[l][i + 1, j, k] == FLUID:
                    self.Adiag[l][i, j, k] += scale
                    self.Ax[l][i, j, k] = -scale
                elif self.grid_type[l][i + 1, j, k] == AIR:
                    self.Adiag[l][i, j, k] += scale

                if self.grid_type[l][i, j - 1, k] == FLUID:
                    self.Adiag[l][i, j, k] += scale
                if self.grid_type[l][i, j + 1, k] == FLUID:
                    self.Adiag[l][i, j, k] += scale
                    self.Ay[l][i, j, k] = -scale
                elif self.grid_type[l][i, j + 1, k] == AIR:
                    self.Adiag[l][i, j, k] += scale

                if self.grid_type[l][i, j, k - 1] == FLUID:
                    self.Adiag[l][i, j, k] += scale
                if self.grid_type[l][i, j, k + 1] == FLUID:
                    self.Adiag[l][i, j, k] += scale
                    self.Az[l][i, j, k] = -scale
                elif self.grid_type[l][i, j, k + 1] == AIR:
                    self.Adiag[l][i, j, k] += scale

    def solve(self, max_iters):
        tol = 1e-12

        self.p.fill(0.0)
        self.As.fill(0.0)
        self.s.fill(0.0)
        self.r[0].copy_from(self.b)

        self.reduce(self.r[0], self.r[0])
        init_rTr = self.sum[None]

        print("init rTr = {}".format(init_rTr))

        if init_rTr < tol:
            print("Converged: init rtr = {}".format(init_rTr))
        else:
            # p0 = 0
            # r0 = b - Ap0 = b
            # z0 = M^-1r0
            # self.z.fill(0.0)
            self.v_cycle()

            # s0 = z0
            self.s.copy_from(self.z[0])

            # zTr
            self.reduce(self.z[0], self.r[0])
            old_zTr = self.sum[None]

            iteration = 0

            for i in range(max_iters):
                # alpha = zTr / sAs
                self.compute_As()
                self.reduce(self.s, self.As)
                sAs = self.sum[None]
                self.alpha[None] = old_zTr / sAs

                # p = p + alpha * s
                self.update_p()

                # r = r - alpha * As
                self.update_r()

                # check for convergence
                self.reduce(self.r[0], self.r[0])
                rTr = self.sum[None]
                if rTr < init_rTr * tol:
                    break

                # z = M^-1r
                self.v_cycle()

                self.reduce(self.z[0], self.r[0])
                new_zTr = self.sum[None]

                # beta = zTrnew / zTrold
                self.beta[None] = new_zTr / old_zTr

                # s = z + beta * s
                self.update_s()
                old_zTr = new_zTr
                iteration = i

            print("Converged to {} in {} iterations".format(rTr, iteration))

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.0
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                self.sum[None] += p[i, j, k] * q[i, j, k]

    def v_cycle(self):
        self.z[0].fill(0.0)
        for l in range(self.multigrid_level - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)

            self.r[l + 1].fill(0.0)
            self.z[l + 1].fill(0.0)
            self.restrict(l)

        # solve Az = r on the coarse grid
        for i in range(self.bottom_smoothing):
            self.smooth(self.multigrid_level - 1, 0)
            self.smooth(self.multigrid_level - 1, 1)

        for l in reversed(range(self.multigrid_level - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.i32):
        # phase: red/black Gauss-Seidel phase
        for i, j, k in self.r[l]:
            if self.grid_type[l][i, j, k] == FLUID and (i + j + k) & 1 == phase:
                self.z[l][i, j, k] = (self.r[l][i, j, k] - self.neighbor_sum(
                    self.Ax[l], self.Ay[l], self.Az[l], self.z[l],
                    self.grid_size[0] // (2**l),
                    self.grid_size[1] // (2**l),
                    self.grid_size[2] // (2**l), i, j, k)) / self.Adiag[l][i, j, k]

    @ti.func
    def neighbor_sum(self, Ax, Ay, Az, z, nx, ny, nz, i, j, k):
        res = Ax[(i - 1 + nx) % nx, j, k] * z[
            (i - 1 + nx) % nx, j, k] + Ax[i, j, k] * z[
                (i + 1) % nx, j, k] + Ay[i, (j - 1 + ny) % ny, k] * z[
                    i, (j - 1 + ny) % ny, k] + Ay[i, j, k] * z[
                        i,
                        (j + 1) % ny, k] + Az[i, j, (k - 1 + nz) % nz] * z[
                            i, j,
                            (k - 1 + nz) % nz] + Az[i, j, k] * z[i, j,
                                                               (k + 1) % nz]

        return res

    @ti.kernel
    def restrict(self, l: ti.template()):
        for i, j, k in self.r[l]:
            if self.grid_type[l][i, j, k] == FLUID:
                Az = self.Adiag[l][i, j, k] * self.z[l][i, j, k]
                Az += self.neighbor_sum(self.Ax[l], self.Ay[l], self.Az[l],
                                        self.z[l], self.grid_size[0] // (2**l),
                                        self.grid_size[1] // (2**l),
                                        self.grid_size[2] // (2**l),
                                        i, j, k)
                res = self.r[l][i, j, k] - Az

                self.r[l + 1][i // 2, j // 2, k // 2] += 0.125 * res

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for i, j, k in self.z[l]:
            self.z[l][i, j, k] += self.z[l + 1][i // 2, j // 2, k // 2]

    @ti.kernel
    def compute_As(self):
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                self.As[i, j, k] = self.Adiag[0][i, j, k] * self.s[
                    i, j, k] + self.Ax[0][i - 1, j, k] * self.s[
                        i - 1, j, k] + self.Ax[0][i, j, k] * self.s[
                            i + 1, j, k] + self.Ay[0][i, j - 1, k] * self.s[
                                i, j - 1, k] + self.Ay[0][i, j, k] * self.s[
                                    i, j + 1,
                                    k] + self.Az[0][i, j, k - 1] * self.s[
                                        i, j, k - 1] + self.Az[0][
                                            i, j, k] * self.s[i, j, k + 1]

    @ti.kernel
    def update_p(self):
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                self.p[i, j,
                       k] = self.p[i, j,
                                   k] + self.alpha[None] * self.s[i, j, k]

    @ti.kernel
    def update_r(self):
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                self.r[0][i, j, k] = self.r[0][
                    i, j, k] - self.alpha[None] * self.As[i, j, k]

    @ti.kernel
    def update_s(self):
        for i, j, k in ti.ndrange(self.grid_size[0], self.grid_size[1], self.grid_size[2]):
            if self.cell_type[i, j, k] == FLUID:
                self.s[i, j,
                       k] = self.z[0][i, j,
                                      k] + self.beta[None] * self.s[i, j, k]
