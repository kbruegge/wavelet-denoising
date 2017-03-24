import click
import datetime
import matplotlib.pyplot as plt
import numpy as np
from ctawave.online_wavelet_analysis import Transient
import random
from tqdm import tqdm
plt.style.use('ggplot')


@click.command()
@click.argument('output_file', type=click.Path(exists=False))
def main(output_file):

    transient = Transient(
                    window_duration=datetime.timedelta(seconds=40),
                    slices_per_cube=40,
                    step=datetime.timedelta(seconds=20)
                )

    target_region = [[69.5, 70.5], [-0.5, 0.5]]
    alt_range = np.array(transient.bin_range[0])
    az_range = np.array(transient.bin_range[1])

    t_background = datetime.datetime.utcnow()
    t_signal = t_background

    for i in tqdm(range(40*10000)):

        alt = random.uniform(*alt_range)
        az = random.uniform(*az_range)
        t_background = t_background + datetime.timedelta(seconds=0.0001)

        transient.add_point(t_background, alt, az)

    for i in tqdm(range(40*10000)):

        alt = random.uniform(*alt_range)
        az = random.uniform(*az_range)
        t_background = t_background + datetime.timedelta(seconds=0.0001)

        transient.add_point(t_background, alt, az)

        alt = random.uniform(*target_region[0])
        az = random.uniform(*target_region[1])
        t_signal = t_background + datetime.timedelta(seconds=random.uniform(0.00005, 0.0001))
        transient.add_point(t_signal, alt, az)

    for i in tqdm(range(80*10000)):

        alt = random.uniform(*alt_range)
        az = random.uniform(*az_range)
        t_background = t_background + datetime.timedelta(seconds=0.0001)

        transient.add_point(t_background, alt, az)

    fig, ax = plt.subplots(1)
    # rotate and align the tick labels so they look better
    for i in range(len(transient.trigger_criterion)):
        ax.plot(transient.trigger_criterion_timestamps[i], transient.trigger_criterion[i], '.')

    import matplotlib.dates as mdates
    ax.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
    fig.autofmt_xdate()

    plt.savefig(output_file)


if __name__ == "__main__":
    main()
