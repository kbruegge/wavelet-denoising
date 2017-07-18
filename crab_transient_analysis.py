import matplotlib.pyplot as plt
import pandas as pd
import pywt
import click
import astropy.units as u

from astropy.io import fits
from collections import OrderedDict
from matplotlib import animation
from ctawave.plot import TransientPlotter
from ctawave.denoise import thresholding_3d
from ctawave.toy_models_crab import simulate_steady_source, \
    simulate_steady_source_with_transient
from tqdm import tqdm
plt.style.use('ggplot')


@click.command()
@click.argument('out_file', type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    '--time_steps',
    '-s',
    type=click.INT,
    help='Number of timesteps to simulate',
    default='100'
)
@click.option(
    '--time_steps_bg',
    '-s_bg',
    type=click.INT,
    help='Number of slices for background simulation',
    default='5'
)
@click.option(
    '--time_per_slice',
    '-t',
    type=click.INT,
    help='Measuring time for one slice in s',
    default='30'
)
@click.option(
    '--cmap',
    '-c',
    help='Colormap to use for histograms',
    default='viridis'
)
@click.option(
    '--cu_flare',
    '-cu',
    help='Transient brightness in crab units',
    default=1.0
)
def main(
            out_file,
            cu_flare,
            time_per_slice,
            time_steps,
            time_steps_bg,
            cmap

        ):
    '''
    Use a crab toy model to create a transient appearing in the FoV of another source.
    A steady background is subtracted and denoised using wavelets.
    This script then creates an animated gif of the whole shebang saved under the
    OUT_FILE argument.
    '''

    cta_perf_fits = fits.open('/home/lena/Software/ctools-1.3.0/caldb/data/cta/prod3b/bcf/South_z20_average_100s/irf_file.fits')
    data_A_eff = cta_perf_fits['EFFECTIVE AREA']
    data_ang_res = cta_perf_fits['POINT SPREAD FUNCTION']
    data_bg_rate = cta_perf_fits['BACKGROUND']

    a_eff_cta_south = pd.DataFrame(OrderedDict({"E_TeV": (data_A_eff.data['ENERG_LO'][0] + data_A_eff.data['ENERG_HI'][0])/2, "A_eff": data_A_eff.data['EFFAREA'][0][0]}))
    ang_res_cta_south = pd.DataFrame(OrderedDict({"E_TeV": (data_ang_res.data['ENERG_LO'][0] + data_ang_res.data['ENERG_HI'][0])/2, "Ang_Res": data_ang_res.data['SIGMA_1'][0][0]}))

    # create cubes for steady source and steady source with transient
    cube_steady = simulate_steady_source(6 * u.deg, 6 * u.deg, a_eff_cta_south, data_bg_rate, ang_res_cta_south, num_slices=time_steps_bg, time_per_slice=time_per_slice * u.s)
    cube_with_transient = simulate_steady_source_with_transient(6 * u.deg, 6 * u.deg, 2 * u.deg, 2 * u.deg, a_eff_cta_south, data_bg_rate, ang_res_cta_south, cu_flare, num_slices=time_steps, time_per_slice=time_per_slice * u.s)

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
