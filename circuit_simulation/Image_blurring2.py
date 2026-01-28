import numpy as np
from scipy.ndimage import convolve, gaussian_filter, map_coordinates


def blur_wigner_images(images, mode="none", blurring_size=3, sigma=1.0, noise_fraction=0.05, noise_std=0.1, loss_eta=0.9, phase_std=0.05, x_range=(-4.0, 4.0)):
    """
    Apply one or several augmentations to a batch of Wigner images (vectorized).
    Parameters
    ----------
    images : np.ndarray --> Shape (M, H, W)
    mode : str or list[str] --> One or several augmentations: ["box", "gaussian", "noise", "none"]
    Returns
    -------
    augmented_images : np.ndarray --> Shape (M, H, W)
    """

    # Normalize mode input
    if isinstance(mode, str):
        modes = [mode]
    else:
        modes = list(mode)

    augmented = images.astype(np.float64, copy=True)

    M, H, W = augmented.shape
    xmin, xmax = x_range
    x = np.linspace(xmin, xmax, H)
    X, P = np.meshgrid(x, x, indexing="ij")


    for m in modes:

        # ----- Box blur -----
        # Physical meaning : Finite detector resolution
        if m == "box":
            if blurring_size % 2 == 0:
                raise ValueError("blurring_size must be odd")

            kernel = np.ones((blurring_size, blurring_size), dtype=np.float64) / blurring_size**2
            augmented = convolve(augmented, kernel[None, :, :], mode="reflect")


        # ----- Gaussian blur -----
        # Physical meaning : Optical mode mismatch / finite bandwidth
        elif m == "gaussian":
            augmented = gaussian_filter(augmented, sigma=(0, sigma, sigma), mode="reflect")


        # ----- Optical loss channel ----- (Main noise process)
        # Physical meaning : Optical losses in fiber couplings or imperfections in beamsplitters ---> Mixing with vacuum
        elif m == "loss":
            if not (0.0 < loss_eta <= 1.0):
                raise ValueError("loss_eta must be in (0, 1]")

            sigma_loss = np.sqrt((1.0 - loss_eta) / (2.0 * loss_eta)) # Effective convolution width (vacuum noise injection)

            augmented = gaussian_filter(augmented, sigma=(0, sigma_loss, sigma_loss), mode="reflect",)

        
        # ----- Phase noise -----
        # Physical meaning : Laser phase drift / Thermal fluctuations ---> Random rotation in phase space
        elif m == "phase":
            rng = np.random.default_rng()
            angles = rng.normal(loc=0.0, scale=phase_std, size=M)

            for i in range(M):
                theta = angles[i]
                c, s = np.cos(theta), np.sin(theta)

                Xr = c * X - s * P
                Pr = s * X + c * P

                # Map rotated coordinates back to pixel indices
                ix = (Xr - xmin) / (xmax - xmin) * (H - 1)
                ip = (Pr - xmin) / (xmax - xmin) * (W - 1)

                coords = np.vstack([ix.ravel(), ip.ravel()])
                rotated = map_coordinates(
                    augmented[i], coords, order=1, mode="reflect"
                )

                augmented[i] = rotated.reshape(H, W)


        # ----- Random pixel noise -----
        # Physical meaning : Other noise processes (electronic noise, detection noise, ...)
        elif m == "noise":
            num_pixels = H * W
            num_noisy = int(noise_fraction * num_pixels)

            rng = np.random.default_rng()
            mask = np.zeros((M, num_pixels), dtype=bool)    # Create noise mask

            for i in range(M):
                idx = rng.choice(num_pixels, size=num_noisy, replace=False)
                mask[i, idx] = True

            mask = mask.reshape(M, H, W)
            noise = rng.normal(loc=0.0, scale=noise_std, size=(M, H, W))

            augmented += noise * mask


        # ----- No changes -----
        elif m == "none":
            continue

        else:
            raise ValueError(
                f"Unknown mode '{m}'. "
                "Choose from ['box', 'gaussian', 'loss', 'phase', 'noise', 'none'].")

    return augmented



def renormalize_wigner_images(images, x_range=(-4.0, 4.0)):
    """
    Renormalize a batch of discretized Wigner functions so that
    ∫ W(x,p) dx dp = 1 for each image.
    """

    M, H, W = images.shape
    xmin, xmax = x_range

    dx = (xmax - xmin) / H
    area_element = dx * dx

    normalized_images = images.copy()

    integrals = np.sum(normalized_images, axis=(1, 2)) * area_element
    integrals[integrals == 0.0] = 1.0

    normalized_images /= integrals[:, None, None]

    return normalized_images


params, images = ???

noisy_images = blur_wigner_images(images, mode=["box", "noise"])
noisy_images = renormalize_wigner_images(noisy_images, x_range=(-4.0, 4.0))
noisy_filename = f"noisy_data_{N_SAMPLES}_samples_{IMAGE_DIM}pix_{data_id}.npz"

save_data(params, noisy_images, noisy_filename)
