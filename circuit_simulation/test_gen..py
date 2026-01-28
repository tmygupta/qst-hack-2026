import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from mrmustard.lab import Circuit, SqueezedVacuum, Number
from mrmustard.lab.transformations import BSgate
from tqdm import tqdm
from mrmustard import math as mm_math

# 1. SETUP
# -----------------------------------------------------------------------------
# Force JAX backend
mm_math.change_backend("jax")
jax.config.update("jax_enable_x64", True)
SEED = 42

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



# 2. PURE JAX WIGNER IMPLEMENTATION
# -----------------------------------------------------------------------------
# We reimplement the Wigner function to avoid 'asnumpy' calls inside the JIT.

@partial(jax.jit, static_argnums=(0,))
def genlaguerre_jax(n, alpha, x):
    """
    Computes generalized Laguerre polynomial L_n^alpha(x) using JAX-compatible recurrence.
    """
    l0 = jnp.ones_like(x)
    if n == 0: return l0
    
    l1 = 1 + alpha - x
    if n == 1: return l1
    
    def body(k, val):
        prev, curr = val
        # Recurrence: (k+1) L_{k+1} = (2k + 1 + alpha - x) L_k - (k + alpha) L_{k-1}
        next_poly = ((2 * k + 1 + alpha - x) * curr - (k + alpha) * prev) / (k + 1)
        return (curr, next_poly)

    _, ln = jax.lax.fori_loop(1, n, body, (l0, l1))
    return ln

def wigner_jax(rho, xvec, pvec):
    """
    Pure JAX implementation of Wigner function for a density matrix rho.
    Removes the 'asnumpy' conversion that breaks JAX tracing.
    """
    X, P = jnp.meshgrid(xvec, pvec)
    Q = X + 1j * P
    alpha_grid = Q / 2.0
    abs_alpha_sq = jnp.abs(alpha_grid)**2
    factor_grid = jnp.exp(-2.0 * abs_alpha_sq) * (2.0 / jnp.pi)
    
    D = rho.shape[0]
    W_final = jnp.zeros_like(X, dtype=jnp.float64)
    
    # We iterate through the density matrix elements.
    # Since D is small (cutoff ~4-5), unrolled loops are efficient.
    for m in range(D):
        for n in range(m + 1): 
            k = m - n
            laguerre = genlaguerre_jax(n, float(k), 4.0 * abs_alpha_sq)
            
            # Stable factorial ratio sqrt(n!/m!)
            fact_ratio = jnp.exp(jax.scipy.special.gammaln(n + 1) - jax.scipy.special.gammaln(m + 1))
            core_term = ((-1)**m) * jnp.sqrt(fact_ratio) * (2 * jnp.conj(alpha_grid))**k * laguerre
            
            if m == n:
                W_final += jnp.real(rho[m, n] * core_term)
            else:
                # Add off-diagonal and its conjugate (2 * Real part)
                W_final += 2.0 * jnp.real(rho[m, n] * core_term)
                
    return W_final * factor_grid

# 3. CIRCUIT SIMULATION
# -----------------------------------------------------------------------------

def single_wigner_fx(r, theta, phi, m0, m1, xvec, pvec):
    """
    Simulates a SINGLE circuit.
    """
    # Input State
    s0 = SqueezedVacuum(mode=0, r=r[0], phi=0.0)
    s1 = SqueezedVacuum(mode=1, r=r[1], phi=np.pi/2)
    s2 = SqueezedVacuum(mode=2, r=r[2], phi=0.0)
    input_state = [s0, s1, s2]
    
    # Interferometer
    BS1 = BSgate(modes=[0, 1], theta=theta[0], phi=phi[0])
    BS2 = BSgate(modes=[1, 2], theta=theta[1], phi=phi[1])
    BS3 = BSgate(modes=[0, 1], theta=theta[2], phi=phi[2]) 
    interferometer = BS1 >> BS2 >> BS3
    
    # Contract
    circ = Circuit(input_state) >> interferometer >> Circuit([m0, m1])
    sout = circ.contract().normalize()
    
    # Calculate Wigner using our SAFE implementation
    # .dm().ansatz.array gets the raw JAX array of the density matrix
    wig = wigner_jax(sout.dm().ansatz.array, xvec, pvec)
    
    return wig

# 4. BATCH GENERATOR (The Fix)
# -----------------------------------------------------------------------------

def generate_data(params, pix_res=32, range_val=4.0):
    
    # Grid Setup
    xvec = jnp.linspace(-range_val, range_val, pix_res)
    pvec = jnp.linspace(-range_val, range_val, pix_res)

    # Convert Inputs
    r_all = jnp.array(params['r'])
    theta_all = jnp.array(params['theta'])
    phi_all = jnp.array(params['phi'])
    n_all = params['n'] 
    
    total_samples = r_all.shape[0]
    all_images = np.zeros((total_samples, pix_res, pix_res))
    unique_ns = np.unique(n_all, axis=0)
    
    print(f"Processing {len(unique_ns)} unique patterns with JAX optimization...")
    
    for n_vals in tqdm(unique_ns):
        # 1. Filter indices
        current_n_pair = tuple(map(int, n_vals))
        mask = (n_all[:, 0] == current_n_pair[0]) & (n_all[:, 1] == current_n_pair[1])
        indices = np.where(mask)[0]
        if len(indices) == 0: continue

        # 2. Create Measurement Objects (Python side, safe from JAX errors)
        m0 = Number((0), n=current_n_pair[0]).dual
        m1 = Number((1), n=current_n_pair[1]).dual

        # 3. Define Scanner
        # We use lax.map, which compiles a loop on the GPU.
        # This bypasses the 'pure_callback' vectorization error.
        def scan_body(packed_args):
            return single_wigner_fx(packed_args[0], packed_args[1], packed_args[2], m0, m1, xvec, pvec)

        # 4. JIT Compile the loop
        jit_scanner = jax.jit(lambda r, t, p: jax.lax.map(scan_body, (r, t, p)))

        # 5. Execute
        batch_images = jit_scanner(r_all[indices], theta_all[indices], phi_all[indices])
        
        all_images[indices] = np.array(batch_images)

    return all_images

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