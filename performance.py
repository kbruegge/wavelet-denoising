import numpy as np
import spectrum
import astropy.units as u
from scipy import integrate
from IPython import embed


def interp_ang_res(E_TeV, df_Ang_Res):
    '''
    Interpolates the angular resolution for given energy E_Tev
    '''
    return np.interp(E_TeV, df_Ang_Res.E_TeV, df_Ang_Res.Ang_Res)


def calc_a_eff_factor(df_A_eff, simulation_area):
    '''
    Returns integrated effective area divided by full impact area
    '''

    simulation_area = simulation_area.to(u.m**2)
    integrate_a_eff = integrate.simps(y=df_A_eff.A_eff, x=df_A_eff.E_TeV) * u.m**2
    integrate_a_impact = simulation_area*(df_A_eff.E_TeV.max() - df_A_eff.E_TeV.min())
    return integrate_a_eff/integrate_a_impact


def interp_eff_area(E_TeV, df_A_eff):
    '''
    Interpolates effective area for given energy
    '''
    return np.interp(E_TeV / u.TeV, df_A_eff.E_TeV, df_A_eff.A_eff)


def response(T, N, e_min, e_max, A_eff, simulation_area):
    '''
    Returns array of events from a source with power law distribution folded with effective area of the telescope

    Parameters:
    index: index of the power law spectrum
    e_min: lower energy bound
    e_max: upper energy bound
    N: number of events coming from source
    A_eff: DataFrame containing values for effective area
    '''

    simulation_area = simulation_area.to(u.m**2)
    events = spectrum.random_power(e_min, e_max, N)
    folded_events = []
    if len(events) > 0:
        for e in events:
            a_eff_event = interp_eff_area(e, A_eff)
            ulimite = (a_eff_event * u.m**2) / (simulation_area)

            if(ulimite.value >= np.random.uniform(0, 1)):
                folded_events.append(e/u.TeV)

    return folded_events


def sample_positions_steady_source(x_pos, y_pos, ang_res):
    '''
    Sample position for every particle with given mean position and angular resolution as mean and covariance for normal distribution
    '''
    mean = [x_pos / u.deg, y_pos / u.deg]
    RA = []
    DEC = []
    for r in ang_res:
        cov = [[r**2, 0], [0, r**2]]
        x, y = np.random.multivariate_normal(mean, cov).T
        RA.append(x)
        DEC.append(y)
    return RA, DEC


def sample_positions_background_random(fov_min, fov_max, N):
    '''
    Sample positions for given number of background events from normal distribution
    '''

    RA_bg = np.random.uniform(fov_min / u.deg, fov_max / u.deg, N)
    DEC_bg = np.random.uniform(fov_min / u.deg, fov_max / u.deg, N)
    return RA_bg, DEC_bg


def integrate_background(bkg, obs_time):
    '''
    Integrate background events from IRF.
    fits-file provides background rates in 1/MeV/s/sr, therefore energy units must be adapted!
    There are different IRFs for different observation times and zenith angles. Here the IRF for
    20 deg and 100 s is used.

    Parameters:
    bkg: irf-file containing background cube
    obs_time: observation time per slice
    '''

    delta_energy = (bkg.data['ENERG_HI'][0] - bkg.data['ENERG_LO'][0]) * u.TeV
    delta_x = (bkg.data['DETX_HI'][0] - bkg.data['DETX_LO'][0]) * u.deg
    delta_y = (bkg.data['DETY_HI'][0] - bkg.data['DETY_LO'][0]) * u.deg
    delta_energy, delta_y, delta_x = np.meshgrid(delta_energy, delta_y, delta_x, indexing='ij')
    bin_volume = delta_energy.to(u.MeV) * (delta_y * delta_x).to(u.sr)
    bg = bkg.data['BGD'][0] * (1/(u.MeV * u.s * u.sr))
    integral = bg * bin_volume
    return integral.sum() * obs_time
