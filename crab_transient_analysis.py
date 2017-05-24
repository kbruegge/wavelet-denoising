import matplotlib.pyplot as plt
import pandas as pd
import pywt
import click

from matplotlib import animation
from ctawave.plot import TransientPlotter
from ctawave.denoise import thresholding_3d
from ctawave.toy_models_crab import simulate_steady_source, \
    simulate_steady_source_with_transient
plt.style.use('ggplot')


@click.command()
@click.argument('out_file', type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    '--bins',
    '-b',
    type=click.INT,
    help='Pixels per axis',
    default='80',
              )
@click.option(
    '--time_steps',
    '-s',
    type=click.INT,
    help='Number of timesteps to simulate',
    default='100'
)
@click.option(
    '--time_per_slice',
    '-t',
    type=click.INT,
    help='Measuring time for one slice',
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
            time_per_slice,
            time_steps,
            cmap
        ):
    '''
    Use a toy model to create a transient appearing in the FoV of another source.
    A steady background is subtracted and denoised using wavelets.
    This script then creates an animated gif of the whoe shebang saved under the
    OUT_FILE argument.
    '''

    ang_res_cta_north = pd.read_csv(
        '/home/lena/Dokumente/CTA/CTA-Performance-North-20150511.ASCII_/CTA-Performance-North-Angres.txt',
        names=['E_TeV', 'Ang_Res'],
        skiprows=10,
        sep='\s+',
        index_col=False
        )

    a_eff_cta_north = pd.read_csv(
        '/home/lena/Dokumente/CTA/CTA-Performance-North-20150511.ASCII_/CTA-Performance-North-0.5h-EffArea.txt',
        names=['E_TeV', 'A_eff'],
        skiprows=10,
        sep='\s+',
        index_col=False)

    cube_steady = simulate_steady_source(3, 3, a_eff_cta_north, ang_res_cta_north)
    cube_with_transient = simulate_steady_source_with_transient(3, 3, 1, 1, a_eff_cta_north, ang_res_cta_north)

    # remove mean measured noise from current cube
    cube = cube_with_transient - cube_steady.mean(axis=0)
    coeffs = pywt.swtn(data=cube, wavelet='bior1.3', level=2, start_level=0)

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
    plt.show()


if __name__ == '__main__':
    main()
