import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def blur_wigner_images(images, mode="none", blurring_size=3, sigma=1.0, noise_fraction=0.05, noise_std=0.1):
    """
    Apply one or several augmentations to a batch of Wigner images (vectorized).

    Parameters
    ----------
    images : np.ndarray
        Shape (M, H, W)

    mode : str or list[str]
        One or several augmentations:
        ["box", "gaussian", "noise", "none"]

    Returns
    -------
    augmented_images : np.ndarray
        Shape (M, H, W)
    """

    # Normalize mode input
    if isinstance(mode, str):
        modes = [mode]
    else:
        modes = list(mode)

    augmented = images.astype(np.float64, copy=True)

    for m in modes:

        # --------------------------------------------------
        # Box blur (vectorized)
        # --------------------------------------------------
        if m == "box":
            if blurring_size % 2 == 0:
                raise ValueError("blurring_size must be odd")

            kernel = np.ones((blurring_size, blurring_size), dtype=np.float64) / blurring_size**2
            augmented = convolve(augmented, kernel[None, :, :], mode="reflect")


        # --------------------------------------------------
        # Gaussian blur (vectorized)
        # --------------------------------------------------
        elif m == "gaussian":
            augmented = gaussian_filter(augmented, sigma=(0, sigma, sigma), mode="reflect")


        # --------------------------------------------------
        # Random pixel noise (vectorized)
        # --------------------------------------------------
        elif m == "noise":
            M, H, W = augmented.shape
            num_pixels = H * W
            num_noisy = int(noise_fraction * num_pixels)

            rng = np.random.default_rng()

            # Create noise mask
            mask = np.zeros((M, num_pixels), dtype=bool)

            for i in range(M):
                idx = rng.choice(num_pixels, size=num_noisy, replace=False)
                mask[i, idx] = True

            mask = mask.reshape(M, H, W)

            noise = rng.normal(loc=0.0, scale=noise_std, size=(M, H, W))

            augmented += noise * mask


        # --------------------------------------------------
        # No augmentation
        # --------------------------------------------------
        elif m == "none":
            continue

        else:
            raise ValueError(
                f"Unknown mode '{m}'. "
                "Choose from ['box', 'gaussian', 'noise', 'none']."
            )

    return augmented



noisy_images = blur_wigner_images(images, mode=["box", "noise"])
noisy_filename = f"noisy_data_{N_SAMPLES}_samples_{IMAGE_DIM}pix_{data_id}.npz"

save_data(params, noisy_images, blurred_filename)
