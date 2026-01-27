import numpy as np

from circuit_simulation import save_data

def draw_params(N : int, r_limits=(0,0.8), theta_limits=(0,np.pi/2),
                     phi_limits=(0,2*np.pi)):
    """
    Draw parameters for the quantum optical setup.
    Parameters
    ----------
    N : int, number of samples to draw
    r_limits : tuple, limits for the squeezing parameter r (inclusive)
    theta_limits : tuple, limits for the angle theta (inclusive)
    phi_limits : tuple, limits for the angle phi (exclusive)
    Returns
    -------
    r : np.ndarray, array of shape (N, 9) containing the drawn parameters
    """
    
    rng = np.random.default_rng()
    r = rng.uniform(r_limits[0], r_limits[1], size=(N, 3))
    theta = rng.uniform(theta_limits[0], theta_limits[1], size=(N, 3))
    phi = rng.uniform(phi_limits[0], phi_limits[1], size=(N, 3))
    
    return np.hstack([r, theta, phi])

def save_data(params,images, filename):
    """
    Save the parameters and images to an .npz file.
    Parameters
    ----------
    params : np.ndarray, array of shape (N, 9) containing the parameters
    images : np.ndarray, array of shape (N, H, W) containing the Wigner images
    filename : str, name of the file to save the data to
    """
    np.savez_compressed(filename, params=params, images=images)
    print(f"Data saved to {filename}")

SEED = 42
N_SAMPLES = 1000
IMAGE_DIM = 64
PARAM_LIMS = {
    'r_lims': (0, 0.8),
    'theta_lims': (0, np.pi / 2),
    'phi_lims': (0, 2 * np.pi),
    'n_modes': (0, 3)
}
np.random.seed(SEED)

data_id = "001"
filename = f"data_{N_SAMPLES}_samples_{IMAGE_DIM}pix_{data_id}.npz"

params = draw_params(N_SAMPLES, **PARAM_LIMS)

images = generate_wigner_sample(params, IMAGE_DIM)

save_data(params, images, filename)

print(f"Data generation complete. Data ID: {data_id}")