# Based on code provided in Exercise 4

from os import stat
from typing import Tuple
from numpy.core.fromnumeric import shape
import taichi as ti
from taichi.misc.gui import rgb_to_hex
import numpy as np
from taichi.lang.ops import cos, sqrt
import utils

ti.init(arch=ti.cpu)

# --------------- Exercise start ---------------------------------

# default simulation parameters
TIME_STEP = 0.045
WIND = 0.0  # float to set wind strength
VORTICITY = 0.0  # for vorticity confinement
MAC_CORMACK = False  # use MacCormack advection scheme?
MAX_ITER = 1e4  # maximum iterations for Gauss-Seidel
MIN_ACC = 1e-5  # minimum accuracy for Gauss-Seidel
# If this is set the divergence is computed again at the end of each simulation step (may be usefull for debugging)
COMPUTE_FINAL_DIVERGENCE = False
# If this is set the vorticity is computed again at the end of each simulation step (may be usefull for debugging)
COMPUTE_FINAL_VORTICITY = False

# Problem 1


@ti.func
def gauss_seidel_poisson_solver(
    pressure: ti.template(),
    divergence: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
    max_iterations: ti.i32,
    min_accuracy: ti.f32,
    rho: ti.f32,
):
    """This method solves the pressure poisson equation using the Gauss-Seidel method.

    Args:
        pressure (ti.template): pressure (this argument will be modified)
        divergence (ti.template): divergence
        dt (ti.template): time step
        dx (ti.template): cell size (assume square cells)
        max_iterations (ti.i32): maximum iterations to perform
        min_accuracy (ti.f32): minimum required accuracy (stop if achieved)
        rho (ti.f32): rho
    """
    dx2 = dx * dx
    res_x, res_y = pressure.shape

    # run Gauss-Seidel as long as max iterations has not been reached and accuracy is not good enough
    residual = min_accuracy + 1
    iterations = 0
    while iterations < max_iterations and residual > min_accuracy:
        residual = 0.0

        for y in range(1, res_y - 1):
            for x in range(1, res_x - 1):
                b = -divergence[x, y] / dt * rho

                # TODO: update the pressure at (x, y)
                pressure[x, y] = (
                    dx2 * b
                    + pressure[x - 1, y]
                    + pressure[x + 1, y]
                    + pressure[x, y - 1]
                    + pressure[x, y + 1]
                ) / 4.0

        for y in range(1, res_y - 1):
            for x in range(1, res_x - 1):
                b = -divergence[x, y] / dt * rho
                # TODO: compute the residual for cell (x, y)

                cell_residual = 0.0
                cell_residual = (
                    b
                    - (
                        4.0 * pressure[x, y]
                        - pressure[x - 1, y]
                        - pressure[x + 1, y]
                        - pressure[x, y - 1]
                        - pressure[x, y + 1]
                    )
                    / dx2
                )

                residual += cell_residual * cell_residual

        residual = sqrt(residual)
        residual /= (res_x - 2) * (res_y - 2)

        iterations += 1


# Problem 2


@ti.func
def velocity_projection(
    velocity_x: ti.template(),
    velocity_y: ti.template(),
    pressure: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements the velocity projection to make the velocity divergence free

    Args:
        velocity_x (ti.template): velocity in x direction (on MAC Grid!) (this will be modified)
        velocity_y (ti.template): velocity in y direction (on MAC Grid!) (this will be modified)
        pressure (ti.template): pressure
        dt (ti.f32): time step
        dx (ti.f32): cell size (assume square cells)
    """

    res_x, res_y = pressure.shape

    for x, y in ti.ndrange((1, res_x), (1, res_y - 1)):
        # TODO: project the x velocity
        velocity_x[x, y] -= dt * (pressure[x, y] - pressure[x - 1, y]) / dx

    for x, y in ti.ndrange((1, res_x - 1), (1, res_y)):
        # TODO: project the y velocity
        velocity_y[x, y] -= dt * (pressure[x, y] - pressure[x, y - 1]) / dx


# Problem 3


@ti.func
def advect_density(
    d_in: ti.template(),
    d_out: ti.template(),
    vx: ti.template(),
    vy: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements semi lagrangian advection for values at cell centers.

    Args:
        d_in (ti.template): Density to be advected (current values)
        d_out (ti.template): Target buffer for advected densities (new values) (this value will be modified)
        vx (ti.template): x velocity field to be used for advection
        vy (ti.template): y velocity field to be used for advection
        dt (ti.f32): time step
        dx (ti.f32): cell size
    """

    res_x, res_y = d_in.shape

    for y in range(1, res_y - 1):
        for x in range(1, res_x - 1):

            # TODO: get velocity at current position
            last_velocity_x = (vx[x, y] + vx[x + 1, y]) / 2.0
            last_velocity_y = (vy[x, y] + vy[x, y + 1]) / 2.0

            # TODO: compute last position (in grid coordinates) using current velocity
            last_position_x = x - dt / dx * last_velocity_x
            last_position_y = y - dt / dx * last_velocity_y

            # make sure last postion is inside grid
            last_position_x = min(max(last_position_x, 1), res_x - 2)
            last_position_y = min(max(last_position_y, 1), res_y - 2)

            # compute cell corners
            x_low, y_low = int(last_position_x), int(last_position_y)
            x_high, y_high = x_low + 1, y_low + 1

            # compute weights for interpolation
            x_weight = last_position_x - x_low
            y_weight = last_position_y - y_low

            # TODO: compute density with bilinear interpolation
            d_out[x, y] = (
                x_weight * y_weight * d_in[x_high, y_high]
                + (1.0 - x_weight) * y_weight * d_in[x_low, y_high]
                + x_weight * (1.0 - y_weight) * d_in[x_high, y_low]
                + (1.0 - x_weight) * (1.0 - y_weight) * d_in[x_low, y_low]
            )


@ti.func
def advect_velocity_x(
    vx_in: ti.template(),
    vx_out: ti.template(),
    vx: ti.template(),
    vy: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements semi lagrangian advection for values in x direction at left cell edge.

    Args:
        vx_in (ti.template): x velocity to be advected (current values)
        vx_out (ti.template): Target buffer for advected x velocity (new values) (this value will be modified)
        vx (ti.template): x velocity field to be used for advection
        vy (ti.template): y velocity field to be used for advection
        dt (ti.f32): time step
        dx (ti.f32): cell size
    """

    res_x, res_y = vx_in.shape
    for y in range(1, res_y - 1):
        for x in range(1, res_x - 1):

            # TODO: get velocity at current position
            last_x_velocity = vx[x, y]
            last_y_velocity = (
                vy[x, y] + vy[x - 1, y] + vy[x - 1, y + 1] + vy[x, y + 1]
            ) / 4.0

            # TODO: compute last position (in grid coordinates) using current velocities
            last_position_x = x - dt / dx * last_x_velocity
            last_position_y = y - dt / dx * last_y_velocity

            # make sure last postion is inside grid
            last_position_x = min(max(last_position_x, 1.5), res_x - 2.5)
            last_position_y = min(max(last_position_y, 1.5), res_y - 2.5)

            # compute cell corners
            x_low, y_low = int(last_position_x), int(last_position_y)
            x_high, y_high = x_low + 1, y_low + 1

            # compute weights for interpolation
            x_weight = last_position_x - x_low
            y_weight = last_position_y - y_low

            # TODO: compute velocity with bilinear interpolation
            vx_out[x, y] = (
                x_weight * y_weight * vx_in[x_high, y_high]
                + (1.0 - x_weight) * y_weight * vx_in[x_low, y_high]
                + x_weight * (1.0 - y_weight) * vx_in[x_high, y_low]
                + (1.0 - x_weight) * (1.0 - y_weight) * vx_in[x_low, y_low]
            )


@ti.func
def advect_velocity_y(
    vy_in: ti.template(),
    vy_out: ti.template(),
    vx: ti.template(),
    vy: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements semi lagrangian advection for values in y direction at bottom cell edge.

    Args:
        vx_in (ti.template): y velocity to be advected (current values)
        vx_out (ti.template): Target buffer for advected y velocity (new values) (this value will be modified)
        vx (ti.template): x velocity field to be used for advection
        vy (ti.template): y velocity field to be used for advection
        dt (ti.f32): time step
        dx (ti.f32): cell size
    """

    res_x, res_y = vy_in.shape

    # velocity y component
    for y in range(1, res_y - 1):
        for x in range(1, res_x - 1):

            # TODO: get velocity at current position
            last_x_velocity = (
                vx[x, y] + vx[x + 1, y] + vx[x + 1, y - 1] + vx[x, y - 1]
            ) / 4.0
            last_y_velocity = vy[x, y]

            # TODO: compute last position (in grid cooridantes) using current velocities
            last_position_x = x - dt / dx * last_x_velocity
            last_position_y = y - dt / dx * last_y_velocity

            # make sure last postion is inside grid
            last_position_x = min(max(last_position_x, 1.5), res_x - 2.5)
            last_position_y = min(max(last_position_y, 1.5), res_y - 2.5)

            # compute cell corners
            x_low, y_low = int(last_position_x), int(last_position_y)
            x_high, y_high = x_low + 1, y_low + 1

            # compute weights for interpolation
            x_weight = last_position_x - x_low
            y_weight = last_position_y - y_low

            # TODO: compute velocity with bilinear interpolation
            vy_out[x, y] = (
                x_weight * y_weight * vy_in[x_high, y_high]
                + (1.0 - x_weight) * y_weight * vy_in[x_low, y_high]
                + x_weight * (1.0 - y_weight) * vy_in[x_high, y_low]
                + (1.0 - x_weight) * (1.0 - y_weight) * vy_in[x_low, y_low]
            )


# Problem 4


@ti.func
def mac_cormack_update(
    d: ti.template(),
    vx: ti.template(),
    vy: ti.template(),
    d_backward: ti.template(),
    d_old: ti.template(),
    vx_backward: ti.template(),
    vx_old: ti.template(),
    vy_backward: ti.template(),
    vy_old: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements the MacCormack advection scheme

    Args:
        d (ti.template): density after standard SL advection (this will be modified)
        vx (ti.template): velocity in x direction after standard SL advection (this will be modified)
        vy (ti.template): velocity in y direction after standard SL advection (this will be modified)
        d_backward (ti.template): buffer to store backward advected density (this may be modified)
        d_old (ti.template): density from before standard SL advection
        vx_backward (ti.template): buffer to store backward advected x velocity (this may be modified)
        vx_old (ti.template): x velocity from before standard SL advection
        vy_backward (ti.template): buffer to store backward advected y velocity (this may be modified)
        vy_old (ti.template): y velocity from before standard SL advection
        dt (ti.f32): time step
        dx (ti.f32): cell size (assume square cells)
    """

    # TODO: advect density and velocities back. Hint: you should make use of the SL advection methods you implemented in Problem 3.
    advect_density(d, d_backward, vx_old, vy_old, -dt, dx)
    advect_velocity_x(vx, vx_backward, vx_old, vy_old, -dt, dx)
    advect_velocity_y(vy, vy_backward, vx_old, vy_old, -dt, dx)

    # do MacCormack update

    for x, y in ti.ndrange(*d.shape):
        # TODO: modify density (d)
        d[x, y] += 0.5 * (d_old[x, y] - d_backward[x, y])

    for x, y in ti.ndrange(*vx.shape):
        # TODO: modify x velocity (vx)
        vx[x, y] += 0.5 * (vx_old[x, y] - vx_backward[x, y])

    for x, y in ti.ndrange(*vy.shape):
        # TODO: modify y velocity (vy)
        vy[x, y] += 0.5 * (vy_old[x, y] - vy_backward[x, y])


@ti.func
def mac_cormack_clamp_density(
    d_in: ti.template(),
    d_old: ti.template(),
    d_forward: ti.template(),
    vx: ti.template(),
    vy: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements the MacCormack clamping operation for values in the cell center.

    Args:
        d_in (ti.template): Density to be clamped (this will be modified)
        d_old (ti.template): Original density before advection
        d_forward (ti.template): New density after SL advection
        vx (ti.template): x velocity used for old cell lookup
        vy (ti.template): y velocity used for old cell lookup
        dt (ti.f32): time step
        dx (ti.f32): cell size
    """

    res_x, res_y = d_in.shape
    for x, y in ti.ndrange((1, res_x - 1), (1, res_y - 1)):

        # TODO: get velocity at current position
        last_x_velocity = (vx[x, y] + vx[x + 1, y]) / 2.0
        last_y_velocity = (vy[x, y] + vy[x, y + 1]) / 2.0

        # TODO: compute last position (in grid coordinates) using current velocities
        last_position_x = x - dt / dx * last_x_velocity
        last_position_y = y - dt / dx * last_y_velocity

        # make sure last postion is inside grid
        last_position_x = min(max(last_position_x, 1), res_x - 2)
        last_position_y = min(max(last_position_y, 1), res_y - 2)

        # compute cell corners
        x_low, y_low = int(last_position_x), int(last_position_y)
        x_high, y_high = x_low + 1, y_low + 1

        # compute min and max old densities at cell corners
        d_min = min(
            d_old[x_low, y_low],
            d_old[x_low, y_high],
            d_old[x_high, y_low],
            d_old[x_high, y_high],
        )
        d_max = max(
            d_old[x_low, y_low],
            d_old[x_low, y_high],
            d_old[x_high, y_low],
            d_old[x_high, y_high],
        )

        # TODO: clamp density
        if d_in[x, y] < d_min or d_in[x, y] > d_max:
            d_in[x, y] = d_forward[x, y]


@ti.func
def mac_cormack_clamp_velocity_x(
    vx_in: ti.template(),
    vx_old: ti.template(),
    vx_forward: ti.template(),
    vx: ti.template(),
    vy: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements the MacCormack clamping operation for values in x direction at left cell edge.

    Args:
        d_in (ti.template): x velocity to be clamped (this will be modified)
        d_old (ti.template): Original x velocity before advection
        d_forward (ti.template): New x velocity after SL advection
        vx (ti.template): x velocity used for old cell lookup
        vy (ti.template): y velocity used for old cell lookup
        dt (ti.f32): time step
        dx (ti.f32): cell size
    """

    res_x, res_y = vx_in.shape
    for x, y in ti.ndrange((1, res_x - 1), (1, res_y - 1)):

        # TODO: get velocity at current position
        last_x_velocity = vx[x, y]
        last_y_velocity = (
            vy[x, y] + vy[x - 1, y] + vy[x - 1, y + 1] + vy[x, y + 1]
        ) / 4.0

        # TODO: compute last position (in grid coordinates) using current velocities
        last_position_x = x - dt / dx * last_x_velocity
        last_position_y = y - dt / dx * last_y_velocity

        # make sure last postion is inside grid
        last_position_x = min(max(last_position_x, 1.5), res_x - 2.5)
        last_position_y = min(max(last_position_y, 1.5), res_y - 2.5)

        # compute cell corners
        x_low, y_low = int(last_position_x), int(last_position_y)
        x_high, y_high = x_low + 1, y_low + 1

        # compute min and max old densities at cell corners
        d_min = min(
            vx_old[x_low, y_low],
            vx_old[x_low, y_high],
            vx_old[x_high, y_low],
            vx_old[x_high, y_high],
        )
        d_max = max(
            vx_old[x_low, y_low],
            vx_old[x_low, y_high],
            vx_old[x_high, y_low],
            vx_old[x_high, y_high],
        )

        # TODO: clamp density
        if vx_in[x, y] < d_min or vx_in[x, y] > d_max:
            vx_in[x, y] = vx_forward[x, y]


@ti.func
def mac_cormack_clamp_velocity_y(
    vy_in: ti.template(),
    vy_old: ti.template(),
    vy_forward: ti.template(),
    vx: ti.template(),
    vy: ti.template(),
    dt: ti.f32,
    dx: ti.f32,
):
    """This method implements the MacCormack clamping operation for values in y direction at bottom cell edge.

    Args:
        d_in (ti.template): y velocity to be clamped (this will be modified)
        d_old (ti.template): Original y velocity before advection
        d_forward (ti.template): New y velocity after SL advection
        vx (ti.template): x velocity used for old cell lookup
        vy (ti.template): y velocity used for old cell lookup
        dt (ti.f32): time step
        dx (ti.f32): cell size
    """

    res_x, res_y = vy_in.shape
    for x, y in ti.ndrange((1, res_x - 1), (1, res_y - 1)):

        # TODO: get velocity at current position
        last_x_velocity = (
            vx[x, y] + vx[x + 1, y] + vx[x + 1, y - 1] + vx[x, y - 1]
        ) / 4.0
        last_y_velocity = vy[x, y]

        # TODO: compute last position (in grid coordinates) using current velocities
        last_position_x = x - dt / dx * last_x_velocity
        last_position_y = y - dt / dx * last_y_velocity

        # make sure last postion is inside grid
        last_position_x = min(max(last_position_x, 1.5), res_x - 2.5)
        last_position_y = min(max(last_position_y, 1.5), res_y - 2.5)

        # compute cell corners
        x_low, y_low = int(last_position_x), int(last_position_y)
        x_high, y_high = x_low + 1, y_low + 1

        # compute min and max old densities at cell corners
        d_min = min(
            vy_old[x_low, y_low],
            vy_old[x_low, y_high],
            vy_old[x_high, y_low],
            vy_old[x_high, y_high],
        )
        d_max = max(
            vy_old[x_low, y_low],
            vy_old[x_low, y_high],
            vy_old[x_high, y_low],
            vy_old[x_high, y_high],
        )

        # TODO: clamp density
        if vy_in[x, y] < d_min or vy_in[x, y] > d_max:
            vy_in[x, y] = vy_forward[x, y]


@ti.func
def copy_field(f_src: ti.template(), f_dst: ti.template()):
    for x, y in ti.ndrange(*f_dst.shape):
        f_dst[x, y] = f_src[x, y]


# --------------- Exercise end ---------------------------------


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
        dt: float = TIME_STEP,
        # wind_strength: float = WIND,
        # vorticity_strength: float = VORTICITY,
        mac_cormack: bool = MAC_CORMACK,
        gauss_seidel_max_iterations: int = MAX_ITER,
        gauss_seidel_min_accuracy: float = MIN_ACC,
        # compute_final_divergence: bool = COMPUTE_FINAL_DIVERGENCE,
        # compute_final_vorticity: bool = COMPUTE_FINAL_VORTICITY,
        resolution: Tuple[float, float, float] = (128, 128, 128),
        paused: bool = True,
    ):

        # parameters that can be changed
        self.dt = dt
        # body force (gravity)
        self.g = ti.Vector([0.0, 0.0, -9.8])
        # self.wind_strength = wind_strength
        # self.vorticity_strength = vorticity_strength
        self.mac_cormack = mac_cormack
        self.gauss_seidel_max_iterations = int(gauss_seidel_max_iterations)
        self.gauss_seidel_min_accuracy = gauss_seidel_min_accuracy
        # self.compute_final_divergence = compute_final_divergence
        # self.compute_final_vorticity = compute_final_vorticity
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

        # self.force_x = ti.field(
        #     ti.f32, shape=(self.resolution[0] + 1, self.resolution[1])
        # )
        # self.force_y = ti.field(
        #     ti.f32, shape=(self.resolution[0], self.resolution[1] + 1)
        # )
        # self.force_z = ti.field(
        #     ti.f32, shape=(self.resolution[0], self.resolution[1], self.resolution[2] + 1)
        # )

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
        # clear_field(self.vorticity)
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
        # clear_field(self.force_x)
        # clear_field(self.force_y)

        # add initial density
        self.apply_density_source()

        # self.r_wind.fill(0)

    def step(self):
        if self.paused:
            return
        self.advance(
            float(self.dt),
            float(self.wind_strength),
            float(self.vorticity_strength),
            1 if self.mac_cormack else 0,
            int(self.gauss_seidel_max_iterations),
            float(self.gauss_seidel_min_accuracy),
            1 if self.compute_final_divergence else 0,
            1 if self.compute_final_vorticity else 0,
        )
        self.t += self.dt
        self.cur_step += 1

    @ti.kernel
    def advance(
        self,
        dt: ti.f32,
        # wind_strength: ti.f32,
        # vorticity_strength: ti.f32,
        mac_cormack: ti.i32,
        gauss_seidel_max_iterations: ti.i32,
        gauss_seidel_min_accuracy: ti.f32,
        compute_final_divergence: ti.i32,
        compute_final_vorticity: ti.i32,
    ):

        # self.apply_density_source()

        # reset forces on grid
        # self.reset_force()

        # sum up all forces
        # self.add_buoyancy()
        # if wind_strength != 0:
        #     self.add_wind(wind_strength)
        # if vorticity_strength > 0:
        #     self.add_vorticity_confinement(vorticity_strength)

        # Apply body force
        self.apply_force

        # particle to grid
        self.p2g()

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

    # desity source

    @ti.func
    def apply_density_source(self):
        x_min = int(np.floor(0.45 * self.resolution[0]))
        x_max = int(np.ceil(0.55 * self.resolution[0]))
        y_min = int(np.floor(0.1 * self.resolution[1]))
        y_max = int(np.ceil(0.15 * self.resolution[1]))
        for x, y in ti.ndrange((x_min, x_max), (y_min, y_max)):
            self.density[x, y] = 1.0

    # # forces

    # @ti.func
    # def add_buoyancy(self):
    #     scaling = 64.0 / self.resolution[0]
    #     for x, y in ti.ndrange(self.resolution[0], self.resolution[1] - 1):
    #         self.force_y[x, y] += (
    #             0.1 * (self.density[x, y - 1] + self.density[x, y]) / 2.0 * scaling
    #         )

    # @ti.func
    # def add_wind(self, strength: ti.f32):
    #     scaling = 64.0 / self.resolution[0] * strength
    #     self.r_wind[0] += 1

    #     f_wind = (
    #         2e-2 * cos(5e-2 * self.r_wind[0]) * cos(3e-2 * self.r_wind[0]) * scaling
    #     )
    #     for x, y in ti.ndrange(self.resolution[0] + 1, self.resolution[1]):
    #         self.force_x[x, y] += f_wind[0]

    # @ti.func
    # def add_vorticity_confinement(self, vorticity_strength: ti.f32):
    #     self.compute_vorticity()

    #     for x, y in ti.ndrange(
    #         (2, self.resolution[0] - 2), (2, self.resolution[1] - 2)
    #     ):
    #         dwdx = (
    #             (abs(self.vorticity[x + 1, y]) - abs(self.vorticity[x - 1, y]))
    #             / self.dx
    #             * 0.5
    #         )
    #         dwdy = (
    #             (abs(self.vorticity[x, y + 1]) - abs(self.vorticity[x, y - 1]))
    #             / self.dx
    #             * 0.5
    #         )
    #         l = sqrt(dwdx * dwdx + dwdy * dwdy) + 1e-6
    #         fx = vorticity_strength * self.dx * self.vorticity[x, y] * dwdy / l
    #         fy = vorticity_strength * self.dx * self.vorticity[x, y] * -dwdx / l
    #         self.force_x[x, y] += fx
    #         self.force_y[x, y] += fy

    @ti.func
    def apply_force(self, dt: ti.f32):
        # for x, y in ti.ndrange(self.resolution[0], self.resolution[1]):
        #     self.velocity_x[x, y] += dt * self.force_x[x, y]
        #     self.velocity_y[x, y] += dt * self.force_y[x, y]
        # only consider body force (gravity) for now
        self.velocity_x += dt * self.
        

    @ti.func
    def reset_force(self):
        clear_field(self.force_x)
        clear_field(self.force_y)

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


if __name__ == "__main__":
    SimulationGUI(HybridSimulator()).run()
