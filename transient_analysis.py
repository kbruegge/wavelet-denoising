import matplotlib.pyplot as plt
from matplotlib import animation
# from tqdm import tqdm
from ctawave.plot import TransientPlotter
from ctawave.denoise import thresholding_3d
from ctawave.toy_models import simulate_steady_source, \
    simulate_steady_source_with_transient, \
    transient_gaussian
import pywt
import click
plt.style.use('ggplot')


@click.command()
@click.argument('out_file', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--bins',
              '-b',
              type=click.INT,
              help='Pixels per axis',
              default='80',
              )
@click.option(
    '--signal_events',
    '-s',
    type=click.INT,
    help='Number of events the steady source emits per image',
    default='100'
)
@click.option(
    '--background_events',
    '-b',
    type=click.INT,
    help='Number of background_events per image',
    default='1000'
)
@click.option(
    '--max_transient_events',
    '-t',
    type=click.INT,
    help='Number of events the transient source emits at its peak activity',
    default='200'
)
@click.option(
    '--time_steps',
    '-ts',
    type=click.INT,
    help='Number of timesteps to simulate',
    default='100'
)
@click.option(
    '--cmap',
    '-c',
    help='Colormap to use for histograms',
    default='viridis'
)
def main(
            out_file,
            bins,
            signal_events,
            background_events,
            max_transient_events,
            time_steps,
            cmap
        ):

    cube_steady = simulate_steady_source(
        num_slices=time_steps,
        source_count=signal_events,
        background_count=background_events,
    )

    def time_dependency():
        return transient_gaussian(time_steps=time_steps, max_events=max_transient_events)

    cube_with_transient = simulate_steady_source_with_transient(
        time_dependency,
        source_count=signal_events,
        background_count=background_events
    )

    # remove mean measured noise from current cube
    cube = cube_with_transient - cube_steady.mean(axis=0)
    coeffs = pywt.swtn(data=cube, wavelet='bior1.3', level=2,)

    # remove noisy coefficents.
    ct = thresholding_3d(coeffs, k=30)
    cube_smoothed = pywt.iswtn(coeffs=ct, wavelet='bior1.3')

    # some Criterion which could be used to trigger this.
    trans_factor = cube_smoothed.max(axis=1).max(axis=1)

    p = TransientPlotter(cube_with_transient,
                         cube_smoothed,
                         trans_factor,
                         cmap=cmap,
                         )

    print('Plotting animation. (Be patient)')
    anim = animation.FuncAnimation(
        p.fig,
        p.step,
        frames=time_steps,
        interval=15,
        blit=True,
    )

    anim.save(out_file, writer='imagemagick', fps=25)
    # plt.show()


if __name__ == '__main__':
    main()
