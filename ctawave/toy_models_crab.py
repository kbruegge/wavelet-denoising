
import numpy as np
import astropy.units as u
from scipy import integrate, signal
from tqdm import tqdm


def random_power(index, e_min, e_max, N):
    '''
    Returns random numbers from power law with given index and energy range (e_min, e_max in TeV)
    '''
    u = np.random.uniform(0, 1, N)
    return ((e_max**(1-index) - e_min**(1-index)) * u + e_min**(1-index))**(1/(1-index))


def number_particles(A, C, index, T, e_min, e_max):
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
    e_min = e_min / u.TeV
    e_max = e_max / u.TeV
    return int((A*T*C/(1-index))*(e_max**(1-index) - e_min**(1-index)) * u.TeV)


def interp_ang_res(E_TeV, df_Ang_Res):
    '''
    Interpolates the angular resolution for given energy E_Tev
    '''
    return np.interp(E_TeV, df_Ang_Res.E_TeV, df_Ang_Res.Ang_Res)


def calc_a_eff_factor(df_A_eff, cta_radius):
    '''
    Returns integrated effective area divided by full impact area
    '''
    full_area = 2*np.pi*(2*cta_radius)**2
    integrate_a_eff = integrate.simps(y=df_A_eff.A_eff, x=df_A_eff.E_TeV) * u.m**2
    integrate_a_impact = full_area*(df_A_eff.E_TeV.max() - df_A_eff.E_TeV.min())
    return integrate_a_eff/integrate_a_impact


def interp_eff_area(E_TeV, df_A_eff):
    '''
    Interpolates effective area for given energy
    '''
    return np.interp(E_TeV / u.TeV, df_A_eff.E_TeV, df_A_eff.A_eff)


def response(index, e_min, e_max, N, A_eff, sample_factor):
    '''
    Returns array of events from a source with power law distribution folded with effective area of the telescope

    Parameters:
    index: index of the power law spectrum
    e_min: lower energy bound
    e_max: upper energy bound
    N: number of events coming from source
    A_eff: DataFrame containing values for effective area
    sample_factor: integrated effective area divided by impact area
    '''
    events = random_power(index, e_min, e_max, N)
    A_effs = []
    for e in events:
        A_effs.append(interp_eff_area(e, A_eff))
    folded_events = np.random.choice(a=events, p=np.divide(A_effs, sum(A_effs)), size=int(sample_factor*N))
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


def sample_positions_background_random(xy_min, xy_max, N):
    '''
    Sample positions for given number of background events from normal distribution
    '''
    RA_bg = np.random.uniform(xy_min / u.deg, xy_max / u.deg, N)
    DEC_bg = np.random.uniform(xy_min / u.deg, xy_max / u.deg, N)
    return RA_bg, DEC_bg


def simulate_steady_source_with_transient(
            x_pos_steady_source,
            y_pos_steady_source,
            x_pos_transient,
            y_pos_transient,
            df_A_eff,
            df_bg_rate,
            df_Ang_Res,
            num_slices=100,
            time_per_slice=30 * u.s,
            bins=[80, 80],
            E_min=0.015 * u.TeV,
            E_max=100 * u.TeV,
            A_crab=2e10 * u.cm**2,
            C_crab=2.83e-11 / (u.cm**2 * u.s * u.TeV),
            index_crab=2.62,
            index_backround=2.7,
            fov_min=0 * u.deg,
            fov_max=12 * u.deg,
            opening_angle=6 * u.deg,
            radius_cta_south=800 * u.m
            ):

    N_steady_source = number_particles(A_crab, C_crab, index_crab, time_per_slice, E_min, E_max)
    N_background_cta = int(df_bg_rate.bg_rate.sum() / (u.s) * time_per_slice)

    sample_factor_a_eff = calc_a_eff_factor(df_A_eff, radius_cta_south)

    N_transient_max = 2*N_steady_source                                         # Random number for transient sample!!!
    transient_scale = (N_transient_max*signal.gaussian(num_slices, std=5)).astype(int)  # arbitrary value for std!!

    slices = []
    for i in tqdm(range(num_slices)):
        folded_events_crab = response(index_crab, E_min, E_max, N_steady_source, df_A_eff, sample_factor_a_eff)
        ang_res_steady_source = interp_ang_res(folded_events_crab, df_Ang_Res)
        RA_crab, DEC_crab = sample_positions_steady_source(x_pos_steady_source, y_pos_steady_source, ang_res_steady_source)
        RA_bg, DEC_bg = sample_positions_background_random(fov_min, fov_max, int(N_background_cta))
        if transient_scale[i] > 0:
            folded_events_transient = response(index_crab, E_min, E_max, transient_scale[i], df_A_eff, sample_factor_a_eff)
            ang_res_transinet = interp_ang_res(folded_events_transient, df_Ang_Res)
            RA_tr, DEC_tr = sample_positions_steady_source(x_pos_transient, y_pos_transient, ang_res_transinet)
        else:
            RA_tr, DEC_tr = [], []
        RA = np.concatenate([RA_bg, RA_tr, RA_crab])
        DEC = np.concatenate([DEC_bg, DEC_tr, DEC_crab])

        slices.append(np.histogram2d(RA, DEC, range=[[fov_min / u.deg, fov_max / u.deg], [fov_min / u.deg, fov_max / u.deg]], bins=bins)[0])

    return np.array(slices)


def simulate_steady_source(
            x_pos,
            y_pos,
            df_A_eff,
            df_bg_rate,
            df_Ang_Res,
            num_slices=100,
            time_per_slice=30 * u.s,
            bins=[80, 80],
            E_min=0.015 * u.TeV,
            E_max=100 * u.TeV,
            A_crab=2e10 * u.cm**2,
            C_crab=2.83e-11 / (u.cm**2 * u.s * u.TeV),
            index_crab=2.62,
            index_backround=2.7,
            fov_min=0 * u.deg,
            fov_max=12 * u.deg,
            opening_angle=6 * u.deg,
            radius_cta_south=800 * u.m
        ):

    N_steady_source = number_particles(A_crab, C_crab, index_crab, time_per_slice, E_min, E_max)
    N_background_cta = int(df_bg_rate.bg_rate.sum() / (u.s) * time_per_slice)

    sample_factor_a_eff = calc_a_eff_factor(df_A_eff, radius_cta_south)

    print(N_steady_source, N_background_cta, sample_factor_a_eff)

    slices = []
    for i in tqdm(range(num_slices)):
        folded_events_crab = response(index_crab, E_min, E_max, N_steady_source, df_A_eff, sample_factor_a_eff)
        ang_res_steady_source = interp_ang_res(folded_events_crab, df_Ang_Res)

        RA_crab, DEC_crab = sample_positions_steady_source(x_pos, y_pos, ang_res_steady_source)
        RA_bg, DEC_bg = sample_positions_background_random(fov_min, fov_max, int(N_background_cta))
        RA = np.concatenate([RA_bg, RA_crab])
        DEC = np.concatenate([DEC_bg, DEC_crab])

        slices.append(np.histogram2d(RA, DEC, bins=bins)[0])

    return np.array(slices)
