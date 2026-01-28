import numpy as np
import jax
import jax.numpy as jnp
from mrmustard.lab import SqueezedVacuum, BSgate, Circuit, Number
from mrmustard import math as mm_math
from mrmustard.physics.wigner import wigner_discretized
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

SEED = 42
mm_math.change_backend("jax")
jax.config.update("jax_enable_x64", True)

def draw_params(N_base: int, r_lims=(0,0.8), theta_lims=(0,np.pi/2),
                     phi_lims=(0,2*np.pi), n_max=3):
    """
    Draw parameters for the quantum optical setup.
    Parameters
    ----------
    N_base : int, number of samples to draw
    r_lims : tuple, limits for the squeezing parameter r (inclusive)
    theta_lims : tuple, limits for the angle theta (inclusive)
    phi_lims : tuple, limits for the angle phi (exclusive)
    n_max : int, maximum photon number (inclusive)
    Returns
    -------
    Let C be the number of permutations of (n0, n1) where n0 and n1
    params : dict, containing the following keys:
        'r': ndarray of shape (C * N_base, 3) - squeezing parameters
        'theta': ndarray of shape (C * N_base, 3) - angle parameters
        'phi': ndarray of shape (C * N_base, 3) - phase parameters
        'n': ndarray of shape (C * N_base, 2) - photon number parameters (n0, n1)
    """
    
    rng = np.random.default_rng(SEED)

    r = rng.uniform(*r_lims, size=(N_base, 3))
    theta = rng.uniform(*theta_lims, size=(N_base, 3))
    phi = rng.uniform(*phi_lims, size=(N_base, 3))
    
    n_values = np.arange(0, n_max + 1)
    # Cartesian product of detection outcomes (e.g. [0,0], [0,1]... [3,3])
    n_grid = np.array(np.meshgrid(n_values, n_values)).T.reshape(-1, 2)
    
    n_combs = len(n_grid)
    
    # Repeat params: (N_base, 3) -> (N_base * n_combs, 3)
    r_batch = np.repeat(r, n_combs, axis=0)
    theta_batch = np.repeat(theta, n_combs, axis=0)
    phi_batch = np.repeat(phi, n_combs, axis=0)
    
    # Tile measurements: (n_combs, 2) -> (N_base * n_combs, 2)
    n_batch = np.tile(n_grid, (N_base, 1))
    
    return {'r': r_batch, 'theta': theta_batch, 'phi': phi_batch, 'n': n_batch}

def single_wigner_fx(r, theta, phi, m0, m1, xvec, pvec):
    """
    Simulates a single quantum optical circuit and returns its Wigner function.
    
    Args:
        r: shape (3,) - squeezing magnitudes
        theta: shape (3,) - BS angles
        phi: shape (3,) - BS phases
        m0: int, photon count for measurement on mode 0
        m1: int, photon count for measurement on mode 1
        xvec: ndarray, values for the phase space grid in x
        pvec: ndarray, values for the phase space grid in p
    """
    
    s0 = SqueezedVacuum(mode=0, r=r[0], phi=0.0)
    s1 = SqueezedVacuum(mode=1, r=r[1], phi=np.pi/2)
    s2 = SqueezedVacuum(mode=2, r=r[2], phi=0.0)
    input_state = [s0, s1, s2]
    
    BS1 = BSgate((0, 1), theta[0], phi[0])
    BS2 = BSgate((1, 2), theta[1], phi[1])
    BS3 = BSgate((0, 1), theta[2], phi[2]) # TODO: check (0,2) or (0,1)
    interferometer = BS1 >> BS2 >> BS3
    
    measurements = [m0, m1]
    
    circ = Circuit(input_state) >> interferometer >> Circuit(measurements)
    
    sout = circ.contract().normalize()
    
    wig, _, _ = wigner_discretized(sout.dm().ansatz.array, xvec, pvec)
    
    return wig

# def generate_data(params, pix_res=15, range_val=4.0):
    """
    Generates Wigner functions by grouping samples by photon number n.
    """
    xvec = jnp.linspace(-range_val, range_val, pix_res)
    pvec = jnp.linspace(-range_val, range_val, pix_res)

    vectorized_wig = jax.vmap(single_wigner_fx, in_axes=(0, 0, 0, None, None, None, None))
    jit_wig_kernel = jax.jit(vectorized_wig)

    r_all = params['r']
    theta_all = params['theta']
    phi_all = params['phi']
    n_all = params['n']
    
    total_samples = r_all.shape[0]
    all_images = np.zeros((total_samples, pix_res, pix_res))
    unique_ns = np.unique(n_all, axis=0)
    
    print(f"Processing {len(unique_ns)} unique photon detection patterns...")
    
    for n_vals in tqdm(unique_ns):
        # A. Find all samples that match this specific (n0, n1) pair
        # Convert to tuple for clean handling
        current_n_pair = tuple(map(int, n_vals))
        
        # boolean mask for the rows where n matches current_n_pair
        mask = (n_all[:, 0] == current_n_pair[0]) & (n_all[:, 1] == current_n_pair[1])
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
            
        m0 = Number((0), current_n_pair[0]).dual
        m1 = Number((1), current_n_pair[1]).dual
        
        batch_r = jnp.array(r_all[indices])
        batch_theta = jnp.array(theta_all[indices])
        batch_phi = jnp.array(phi_all[indices])
        
        batch_images = jit_wig_kernel(
            batch_r, 
            batch_theta, 
            batch_phi, 
            m0, 
            m1,
            xvec, 
            pvec
        )
        
        all_images[indices] = np.array(batch_images)

    return all_images

def generate_data(params, pix_res=32, range_val=4.0):
    
    # 1. Setup JAX Grid
    xvec = jnp.linspace(-range_val, range_val, pix_res)
    pvec = jnp.linspace(-range_val, range_val, pix_res)

    # 2. Prepare Data Inputs
    r_all = jnp.array(params['r'])
    theta_all = jnp.array(params['theta'])
    phi_all = jnp.array(params['phi'])
    n_all = params['n'] 
    
    total_samples = r_all.shape[0]
    all_images = np.zeros((total_samples, pix_res, pix_res))
    unique_ns = np.unique(n_all, axis=0)
    
    print(f"Processing {len(unique_ns)} unique detection patterns using lax.map...")
    
    for n_vals in tqdm(unique_ns):
        # A. Indices for this group
        current_n_pair = tuple(map(int, n_vals))
        mask = (n_all[:, 0] == current_n_pair[0]) & (n_all[:, 1] == current_n_pair[1])
        indices = np.where(mask)[0]
        
        if len(indices) == 0: continue

        # B. Create Measurement Objects (Constant for this batch)
        # We perform the creation in Python to handle the integer 'cutoff' logic safely
        m0 = Number((0), n=current_n_pair[0], cutoff=current_n_pair[0]+1).dual
        m1 = Number((1), n=current_n_pair[1], cutoff=current_n_pair[1]+1).dual

        # C. Define the Batch Runner with lax.map
        # We define this *inside* the loop because m0/m1 change for each group.
        # This function takes one sample's (r, th, ph) and returns one image.
        def scan_body(packed_args):
            r_s, th_s, ph_s = packed_args
            return single_wigner_fx(r_s, th_s, ph_s, m0, m1, xvec, pvec)

        # D. JIT Compile the Loop
        # lax.map maps 'scan_body' over the leading axis of the inputs.
        # We JIT the entire mapping operation.
        jit_scanner = jax.jit(lambda r, t, p: jax.lax.map(scan_body, (r, t, p)))

        # E. Run Batch
        batch_r = r_all[indices]
        batch_theta = theta_all[indices]
        batch_phi = phi_all[indices]
        
        # This runs the loop on the device
        batch_images = jit_scanner(batch_r, batch_theta, batch_phi)
        
        all_images[indices] = np.array(batch_images)

    return all_images

def save_data(params, images, filename):
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


N_SAMPLES = 10
IMAGE_DIM = 15
PARAM_LIMS = {
    'r_lims': (0, 0.8),
    'theta_lims': (0, np.pi / 2),
    'phi_lims': (0, 2 * np.pi),
    'n_max': 3
}
np.random.seed(SEED)

data_id = "001"
filename = f"../data/data_{N_SAMPLES}_samples_{IMAGE_DIM}pix_{data_id}.npz"

params = draw_params(N_SAMPLES, **PARAM_LIMS)

images = generate_data(params, pix_res=IMAGE_DIM)

save_data(params, images, filename)

print(f"Data generation complete. Data ID: {data_id}")