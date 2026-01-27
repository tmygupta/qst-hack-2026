import numpy as np

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
