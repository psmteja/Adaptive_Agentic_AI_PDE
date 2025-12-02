import numpy as np
import torch
import matplotlib.pyplot as plt

from pdetransformer.core.mixed_channels import PDETransformer


def generate_heat_2d(nx=64, ny=64, nt=10, dt=0.05, nu=0.2):
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u0 = np.exp(-5.0 * (X**2 + Y**2)).astype(np.float32)
    u = np.zeros((nt, nx, ny), dtype=np.float32)
    u[0] = u0

    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)

    for t in range(1, nt):
        un = u[t - 1]
        u_xx = (np.roll(un, -1, axis=0) - 2.0 * un + np.roll(un, 1, axis=0)) / (dx**2)
        u_yy = (np.roll(un, -1, axis=1) - 2.0 * un + np.roll(un, 1, axis=1)) / (dy**2)
        lap = u_xx + u_yy
        u[t] = un + dt * nu * lap

    return u


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PDETransformer.from_pretrained(
        "thuerey-group/pde-transformer",
        subfolder="mc-s",
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded PDE-Transformer mc-s with {n_params:,} parameters")

    # --- make toy PDE data ---
    nt = 5
    nx = ny = 64
    u = generate_heat_2d(nx=nx, ny=ny, nt=nt)
    print("Generated heat equation data with shape:", u.shape)

    t0, t1, t2 = 0, 1, 2
    u_t0 = u[t0]
    u_t1 = u[t1]
    u_t2 = u[t2]

    x_in = np.stack([u_t0, u_t1], axis=0)      # (2, nx, ny)
    x_in = torch.from_numpy(x_in)[None, ...]   # (1, 2, nx, ny)
    x_in = x_in.to(device=device, dtype=torch.float32)

    print("Input tensor to PDE-Transformer:", x_in.shape)

    with torch.no_grad():
        y_out = model(x_in)

    # ---------- HERE'S THE IMPORTANT CHANGE ----------
    print("Raw model output type:", type(y_out))

    if isinstance(y_out, torch.Tensor):
        y = y_out
    elif hasattr(y_out, "prediction"):
        # current pdetransformer versions usually use this
        y = y_out.prediction
    else:
        # fallback: first entry if it's dict-like
        try:
            y = next(iter(y_out.values()))
        except Exception as e:
            raise TypeError(
                f"Don't know how to get tensor from PDEOutput. "
                f"type={type(y_out)}, dir={dir(y_out)}"
            ) from e
    # -------------------------------------------------

    print("Model tensor output shape:", y.shape)

    # Assume channel 1 is "next state"
    y_next = y[0, 1].cpu().numpy()
    gt_next = u_t2

    mse = np.mean((y_next - gt_next) ** 2)
    print(f"MSE between PDE-Transformer prediction and diffusion solver: {mse:.6e}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(u_t0, origin="lower")
    axes[0].set_title("u(t0)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(gt_next, origin="lower")
    axes[1].set_title("Ground truth u(t2)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(y_next, origin="lower")
    axes[2].set_title("PDE-Transformer pred")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
