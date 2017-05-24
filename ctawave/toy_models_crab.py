
import numpy as np
from scipy import integrate, signal
from tqdm import tqdm
# from IPython import embed


def random_power(index, e_min, e_max, N):
    u = np.random.uniform(0, 1, N)
    return ((e_max**(1-index) - e_min**(1-index)) * u + e_min**(1-index))**(1/(1-index))


def N_E(A, C, Omega, index, T, e_min, e_max):
    # T in s
    # A in m^2 or cm^2
    # E in GeV or TeV depends on flux units
    # C + phi_0, f_0, Vorfaktor
    return int((A*T*C*Omega/(1-index))*(e_max**(1-index) - e_min**(1-index)))


def interp_ang_res(E_TeV, df_Ang_Res):
    return np.interp(E_TeV, df_Ang_Res.E_TeV, df_Ang_Res.Ang_Res)


def calc_a_eff_factor(df_A_eff, full_area=2e6):
    integrate_a_eff = integrate.simps(y=df_A_eff.A_eff, x=df_A_eff.E_TeV)
    integrate_a_impact = full_area*(df_A_eff.E_TeV.max() - df_A_eff.E_TeV.min())      # Estimated value for impact area: 2e6 m^2
    return integrate_a_eff/integrate_a_impact


def A_eff_E(E_TeV, df_A_eff):
    return np.interp(E_TeV, df_A_eff.E_TeV, df_A_eff.A_eff)


def folded_spectrum(index, e_min, e_max, N, A_eff, sample_factor):
    events = random_power(index, e_min, e_max, N)
    A_effs = []
    for e in events:
        A_effs.append(A_eff_E(e, A_eff))
    folded_events = np.random.choice(a=events, p=np.divide(A_effs, sum(A_effs)), size=int(sample_factor*N))
    return folded_events


def sample_positions_steady_source(x_pos, y_pos, ang_res):
    mean = [x_pos, y_pos]
    RA = []
    DEC = []
    for r in ang_res:
        cov = [[r, 0], [0, r]]
        x, y = np.random.multivariate_normal(mean, cov).T
        RA.append(x)
        DEC.append(y)
    return RA, DEC


def sample_positions_background_random(xy_min, xy_max, N):
    RA_bg = np.random.uniform(xy_min, xy_max, N)
    DEC_bg = np.random.uniform(xy_min, xy_max, N)
    return RA_bg, DEC_bg


def simulate_steady_source_with_transient(
            x_pos_steady_source,
            y_pos_steady_source,
            x_pos_transient,
            y_pos_transient,
            df_A_eff,
            df_Ang_Res,
            num_slices=100,
            time_per_slice=30,
            bins=[80, 80],
            E_min=0.015,
            E_max=100,
            A_crab=2e10,
            A_background=2e6,
            C_crab=2.83e-11,       # 1/(cm^2 s TeV)
            C_background=1.8e4,    # 1/(m^2 s sr GeV)
            index_crab=2.62,
            index_backround=2.7,
            fov_min=0,
            fov_max=6,
            opening_angle=3
            ):

    Omega = 2*np.pi*(1-np.cos(opening_angle*np.pi/180))
    N_steady_source = N_E(A_crab, C_crab, 1, index_crab, time_per_slice, E_min, E_max)
    N_background = N_E(A_background, C_background, Omega, index_backround, time_per_slice, E_min*1e3, E_max*1e3)

    sample_factor_a_eff = calc_a_eff_factor(df_A_eff)

    false_positive = 0.001                                                      # Estimated fp-rate! (Diss. Temme)

    N_transient_max = 2*N_steady_source                                         # Random number for transient sample!!!
    transient_scale = (N_transient_max*signal.gaussian(num_slices, std=5)).astype(int)  # arbitrary value for std!!

    slices = []
    for i in tqdm(range(num_slices)):
        folded_events_crab = folded_spectrum(index_crab, E_min, E_max, N_steady_source, df_A_eff, sample_factor_a_eff)
        ang_res_steady_source = interp_ang_res(folded_events_crab, df_Ang_Res)
        RA_crab, DEC_crab = sample_positions_steady_source(x_pos_steady_source, y_pos_steady_source, ang_res_steady_source)
        RA_bg, DEC_bg = sample_positions_background_random(fov_min, fov_max, int(N_background*sample_factor_a_eff*false_positive))
        if transient_scale[i] > 0:
            folded_events_transient = folded_spectrum(index_crab, E_min, E_max, transient_scale[i], df_A_eff, sample_factor_a_eff)
            ang_res_transinet = interp_ang_res(folded_events_transient, df_Ang_Res)
            RA_tr, DEC_tr = sample_positions_steady_source(x_pos_transient, y_pos_transient, ang_res_transinet)
        else:
            RA_tr, DEC_tr = [], []
        RA = np.concatenate([RA_bg, RA_tr, RA_crab])
        DEC = np.concatenate([DEC_bg, DEC_tr, DEC_crab])

        slices.append(np.histogram2d(RA, DEC, range=[[fov_min, fov_max], [fov_min, fov_max]], bins=bins)[0])

    return np.array(slices)


def simulate_steady_source(
            x_pos,
            y_pos,
            df_A_eff,
            df_Ang_Res,
            num_slices=100,
            time_per_slice=30,
            bins=[80, 80],
            E_min=0.015,
            E_max=100,
            A_crab=2e10,
            A_background=2e6,
            C_crab=2.83e-11,       # 1/(cm^2 s TeV)
            C_background=1.8e4,    # 1/(m^2 s sr GeV)
            index_crab=2.62,
            index_backround=2.7,
            fov_min=0,
            fov_max=6,
            opening_angle=3
        ):

    Omega = 2*np.pi*(1-np.cos(opening_angle*np.pi/180))
    N_steady_source = N_E(A_crab, C_crab, 1, index_crab, time_per_slice, E_min, E_max)
    N_background = N_E(A_background, C_background, Omega, index_backround, time_per_slice, E_min*1e3, E_max*1e3)

    sample_factor_a_eff = calc_a_eff_factor(df_A_eff)

    false_positive = 0.001                                                      # Estimated fp-rate! (Diss. Temme)

    slices = []
    for i in tqdm(range(num_slices)):
        folded_events_crab = folded_spectrum(index_crab, E_min, E_max, N_steady_source, df_A_eff, sample_factor_a_eff)
        ang_res_steady_source = interp_ang_res(folded_events_crab, df_Ang_Res)

        RA_crab, DEC_crab = sample_positions_steady_source(x_pos, y_pos, ang_res_steady_source)
        RA_bg, DEC_bg = sample_positions_background_random(fov_min, fov_max, int(N_background*sample_factor_a_eff*false_positive))
        RA = np.concatenate([RA_bg, RA_crab])
        DEC = np.concatenate([DEC_bg, DEC_crab])

        slices.append(np.histogram2d(RA, DEC, bins=bins)[0])

    return np.array(slices)
