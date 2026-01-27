import numpy as np

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
    
    r = np.random.Generator.uniform(r_limits[0], r_limits[1], size=(N, 3))
    theta = np.random.Generator.uniform(theta_limits[0], theta_limits[1], size=(N, 3))
    phi = np.random.Generator.uniform(phi_limits[0], phi_limits[1], size=(N, 3))
    
    return np.hstack([r, theta, phi])
    