
import numpy as np
import astropy.units as u
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm


def background_poissonian(expected_background_events=1000, size=[80, 80]):
    lam = expected_background_events / np.array(size).prod()
    background = np.random.poisson(lam=lam, size=size)
    return background


def signal_gaussian(
            signal_location=np.array([61, 21])*u.deg,
            fov_center=np.array([60, 20])*u.deg,
            width=0.05*u.deg,
            signal_events=80,
            bins=[80, 80],
            fov=4*u.deg,
        ):

    # reshape so if signal_events = 1 the array can be indexed in the same way.
    signal = multivariate_normal.rvs(
                mean=signal_location.value,
                cov=width.value,
                size=signal_events
                ).reshape(signal_events, 2)
    r = np.array([fov_center - fov/2, fov_center + fov/2]).T

    signal_hist, _, _ = np.histogram2d(signal[:, 0], signal[:, 1], bins=bins, range=r)
    return signal_hist


def transient_gaussian(time_steps, width=0.05, max_events=100):
    '''
    simulate a simple transient event. it follows a simple gaussian distirbution
    in time.
    returns two list: time_steps, expected_events
    '''
    t = np.linspace(0, 1, num=time_steps)
    # normalize so that the maximum is at max_events
    g = norm.pdf(t, loc=0.5, scale=width)
    g = (g / g.max()) * max_events

    # and pull poissonian from it
    events = np.random.poisson(lam=g)
    return t, events


def simulate_steady_source_with_transient(
            time_dependend_transient,
            source_count=100,
            background_count=1000,
            bins=[80, 80]
            ):
    '''
    create a cube of images with a steady source and a steady background.
    the 'source_count' and 'background_count' are the total number of events
    per slice in the cube.

    The time_dependend_transient is a function of the form
    time_steps, expected_events = transient()
    '''

    slices = []
    transient_location = np.array([59, 20])*u.deg
    time, values = time_dependend_transient()
    for t, v in tqdm(zip(time, values), total=len(time)):
        H = background_poissonian(
                expected_background_events=background_count,
                size=bins
            ) + signal_gaussian(
                signal_events=source_count,
                bins=bins
            )

        transient_signal = signal_gaussian(
                        signal_location=transient_location,
                        signal_events=v,
                        width=0.01*u.deg,
                        bins=bins,
                        )

        H = H + transient_signal
        slices.append(H)
    return np.array(slices)


def simulate_steady_source(
            num_slices=100,
            source_count=200,
            background_count=1000,
            bins=[80, 80]
        ):
    slices = []
    for i in tqdm(range(num_slices)):
        H = background_poissonian(expected_background_events=background_count, size=bins) \
                + signal_gaussian(signal_events=source_count, bins=bins)

        slices.append(H)
    return np.array(slices)
