import numpy as np
import astropy.units as u
from scipy import integrate, signal
from tqdm import tqdm

def random_power(e_min, e_max, N):
    '''
    Returns random numbers from power law with given index and energy range (e_min, e_max in TeV)
    '''
    index = 2.62
    u = np.random.uniform(0, 1, N)
    return ((e_max**(1-index) - e_min**(1-index)) * u + e_min**(1-index))**(1/(1-index))


def number_particles_crab(T, e_min, e_max):
    '''
    Returns the number of particles arriving from a pointlike source with known power law distribution

    Parameters:
    A: Area
    C: Constant flux factor
    index: power lax index
    T: observation time
    e_min: lower energy bound
    e_max: upper energy bound
    '''
    index = 2.62
    A = 2e10 * u.cm**2
    C = 2.83e-11 / (u.cm**2 * u.s * u.TeV)
    e_min = e_min / u.TeV
    e_max = e_max / u.TeV

    return int((A*T*C/(1-index))*(e_max**(1-index) - e_min**(1-index)) * u.TeV)
