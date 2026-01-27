import numpy as np

def draw_params(N : int, r_limits=(0,0.8), theta_limits=(0,np.pi/2),
                     phi_limits=(0,2*np.pi), n_modes=(0,3)):
    """
    Draw parameters for the quantum optical setup.
    Parameters
    ----------
    N : int, number of samples to draw
    r_limits : tuple, limits for the squeezing parameter r (inclusive)
    theta_limits : tuple, limits for the angle theta (inclusive)
    phi_limits : tuple, limits for the angle phi (exclusive)
    n_modes : tuple, limits for the number of modes (inclusive)
    Returns
    -------
    Let C be the number of permutations of (n0, n1) where n0 and n1
    params : dict, containing the following keys:
        'r': ndarray of shape (C * N, 3) - squeezing parameters
        'theta': ndarray of shape (C * N, 3) - angle parameters
        'phi': ndarray of shape (C * N, 3) - phase parameters
        'n': ndarray of shape (C * N, 2) - photon number parameters (n0, n1)
    """
    
    rng = np.random.default_rng()
    r = rng.uniform(r_limits[0], r_limits[1], size=(N, 3))
    theta = rng.uniform(theta_limits[0], theta_limits[1], size=(N, 3))
    phi = rng.uniform(phi_limits[0], phi_limits[1], size=(N, 3))
    
    base_params = np.hstack([r, theta, phi])  # Shape: (N, 9)
    
    n_values = np.arange(n_modes[0], n_modes[1] + 1)
    n0_grid, n1_grid = np.meshgrid(n_values, n_values, indexing='ij')
    n0_flat = n0_grid.flatten()
    n1_flat = n1_grid.flatten()
    num_combinations = len(n0_flat)
    
    repeated_params = np.repeat(base_params, num_combinations, axis=0)
    
    n0_column = np.tile(n0_flat, N).reshape(-1, 1)
    n1_column = np.tile(n1_flat, N).reshape(-1, 1)
    
    # Group params into arrays: [r_array, theta_array, phi_array, n_array]
    r_vals = repeated_params[:, :3]
    theta_vals = repeated_params[:, 3:6]
    phi_vals = repeated_params[:, 6:]
    n_vals = np.hstack([n0_column, n1_column])
    
    return {'r': r_vals, 'theta': theta_vals, 'phi': phi_vals, 'n': n_vals}

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

images = generate_wigner_sample(**params, IMAGE_DIM)

save_data(params, images, filename)

print(f"Data generation complete. Data ID: {data_id}")