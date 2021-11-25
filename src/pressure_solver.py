import taichi as ti

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

        residual = ti.sqrt(residual)
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