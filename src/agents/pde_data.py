import numpy as np


def generate_heat_data(nx: int = 64,
                       ny: int = 64,
                       nt: int = 10,
                       dt: float = 0.05,
                       nu: float = 0.2) -> np.ndarray:
    """Generate a simple 2D heat (diffusion) evolution.

    PDE:
        du/dt = nu * (d^2u/dx^2 + d^2u/dy^2)

    Uses a crude finite-difference scheme with periodic boundaries.

    Args:
        nx, ny: grid resolution
        nt: number of time steps
        dt: time step size
        nu: diffusion coefficient

    Returns:
        u: array of shape (nt, nx, ny)
    """
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initial condition: Gaussian bump
    u0 = np.exp(-5.0 * (X**2 + Y**2)).astype(np.float32)

    u = np.zeros((nt, nx, ny), dtype=np.float32)
    u[0] = u0

    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    for t in range(1, nt):
        un = u[t - 1]

        # Periodic Laplacian using np.roll (quick & dirty)
        u_xx = (np.roll(un, -1, axis=0) - 2.0 * un + np.roll(un, 1, axis=0)) / (dx**2)
        u_yy = (np.roll(un, -1, axis=1) - 2.0 * un + np.roll(un, 1, axis=1)) / (dy**2)
        lap = u_xx + u_yy

        u[t] = un + dt * nu * lap

    return u
