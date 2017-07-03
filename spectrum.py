import numpy as np
import astropy.units as u
from scipy import integrate, signal
from tqdm import tqdm
from IPython import embed

def random_power(e_min, e_max, N):
    '''
    Returns random numbers from power law with given index and energy range (e_min, e_max in TeV)
    '''
    index = 2.48
    u = np.random.uniform(0, 1, N)
    return ((e_max**(1-index) - e_min**(1-index)) * u + e_min**(1-index))**(1/(1-index))


def number_particles_crab(T, e_min, e_max, simulation_area):
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
#     energy_unit = u.TeV
#     index = 2.62
#     C = 2.83e-11 / (u.cm**2 * u.s * u.TeV)
#     E_0 = 1 * u.TeV

    e_min = e_min.to(u.MeV)
    e_max = e_max.to(u.MeV)
    index = 2.48
    C = 5.7e-16 / (u.cm**2 * u.s * u.MeV)
    E_0 = 0.3e6 * u.MeV

    # cta_radius = (800 * u.m).to(u.cm)
    # A = 2*np.pi*(2*cta_radius)**2
    # A_ctools = (1.2e7 * u.m**2).to(u.cm**2)

    return int((simulation_area.to(u.cm**2)*C*T*E_0**(index)/(1-index))*((e_max)**(1-index) - (e_min)**(1-index)))
