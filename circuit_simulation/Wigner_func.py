import numpy as np
from mrmustard.lab import Circuit, SqueezedVacuum, Number
from mrmustard.lab.transformations import BSgate
from mrmustard.physics.wigner import wigner_discretized


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
    
    # Create (M, 4) array where each row contains 4 sub-arrays
    M = repeated_params.shape[0]
    result = np.empty((M, 4), dtype=object)
    for i in range(M):
        result[i, 0] = r_vals[i]
        result[i, 1] = theta_vals[i]
        result[i, 2] = phi_vals[i]
        result[i, 3] = n_vals[i]
    
    return {'r': r_vals, 'theta': theta_vals, 'phi': phi_vals, 'n': n_vals}


def wigner_from_params(
    r_vals,          # shape (3,)
    theta_vals,      # shape (3,)
    phi_vals,        # shape (3,)
    n_vals,          # shape (2,)
    grid_size=15,
    x_max=4.0
):
    """
    Compute a discretized Wigner function for one parameter set.

    Returns
    -------
    wigner : ndarray of shape (grid_size, grid_size)
    """

    Nl = r_vals.shape[0]
    # Input squeezed states
    input_state = [ SqueezedVacuum(
                                    i,
                                    r_vals[i],
                                    phi=(0 if i % 2 == 1 else np.pi / 2)
                                )
        for i in range(Nl)
    ]

    # Interferometer
    BS1 = BSgate([0, 1], theta_vals[0], phi_vals[0])
    BS2 = BSgate([1, 2], theta_vals[1], phi_vals[1])
    BS3 = BSgate([0, 2], theta_vals[2], phi_vals[2])
    interferometer = BS1 >> BS2 >> BS3

    # Post-selection
    measurement = [
        Number(i, int(n_vals[i])).dual for i in range(Nl - 1)
    ]

    # Circuit contraction
    c = Circuit(input_state) >> interferometer >> Circuit(measurement)
    out = c.contract().normalize()

    # Phase-space grid
    xvec = np.linspace(-x_max, x_max, grid_size)
    pvec = np.linspace(-x_max, x_max, grid_size)

    # Wigner function
    wigner, _, _ = wigner_discretized(
        out.dm().ansatz.array,
        xvec,
        pvec
    )

    return wigner


def generate_wigner_dataset(
    params,          # output dict from draw_params
    grid_size=15,
    x_max=4.0,
    verbose=False
):
    """
    Generate a Wigner array for each parameter set.

    Parameters
    ----------
    params : dict
        Output of draw_params(), with keys 'r', 'theta', 'phi', 'n'
    grid_size : int
        Size of the Wigner grid (grid_size x grid_size)
    x_max : float
        Phase-space bounds [-x_max, x_max]
    verbose : bool
        If True, prints progress information

    Returns
    -------
    wigners : ndarray of shape (M, grid_size, grid_size)
    """

    r_all = params['r']
    theta_all = params['theta']
    phi_all = params['phi']
    n_all = params['n']

    M = r_all.shape[0]

    wigners = np.zeros((M, grid_size, grid_size), dtype=np.float64)

    for i in range(M):
        if verbose and i % 50 == 0:
            print(f"Computing Wigner {i+1}/{M}")

        wigners[i] = wigner_from_params(
            r_vals=r_all[i],
            theta_vals=theta_all[i],
            phi_vals=phi_all[i],
            n_vals=n_all[i],
            grid_size=grid_size,
            x_max=x_max
        )

    return wigners

