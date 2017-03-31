import click
import datetime
import matplotlib.pyplot as plt
import numpy as np
from ctawave.online_wavelet_analysis import Transient
from ctawave.plot import TransientPlotter
import random
from tqdm import tqdm
from scipy.stats import norm
plt.style.use('ggplot')


def background_generator(alt_range=[62.5, 78.5], az_range=[-12.5, 12.5]):
    while True:
        alt = random.uniform(*alt_range)
        az = random.uniform(*az_range)
        yield alt, az


def signal_generator(center=[70, 0], width=1):
    while True:
        yield norm.rvs(loc=center, scale=width)


@click.command()
@click.argument('output_file', type=click.Path(exists=False))
def main(output_file):

    fov_region = [[62.5, 78.5], [-12.5, 12.5]]

    background_per_second = 10000
    signal_per_second = 500
    interval = 30  # seconds

    protons = background_generator()
    gammas = signal_generator()

    transient = Transient(
                    window_duration=datetime.timedelta(seconds=30),
                    slices_per_cube=32,
                    step=datetime.timedelta(seconds=7),
                    bins=[32, 32],
                    bin_range=fov_region,
                )

    t_start = datetime.datetime.utcnow()

    N = background_per_second * interval
    timedeltas = np.sort(np.random.random(N) * interval)
    time_stamps = [t_start + datetime.timedelta(seconds=t) for t in timedeltas]

    for t in tqdm(time_stamps):
        alt, az = next(protons)
        transient.add_point(t, alt, az)

    t_start = max(time_stamps)

    N = (signal_per_second + background_per_second) * interval
    p = np.array([background_per_second * interval, signal_per_second * interval]) / N

    timedeltas = np.sort(np.random.random(N) * interval)
    time_stamps = [t_start + datetime.timedelta(seconds=t) for t in timedeltas]

    for t in tqdm(time_stamps):
        alt, az = next(np.random.choice([protons, gammas], p=p))
        transient.add_point(t, alt, az)

    t_start = max(time_stamps)

    N = background_per_second * interval
    timedeltas = np.sort(np.random.random(N) * interval)
    time_stamps = [t_start + datetime.timedelta(seconds=t) for t in timedeltas]

    for t in tqdm(time_stamps):
        alt, az = next(protons)
        transient.add_point(t, alt, az)

    fig = TransientPlotter.plot_trigger_criterion(transient)
    fig.savefig(output_file)


if __name__ == "__main__":
    main()
