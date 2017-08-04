import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
import click
import astropy.units as u

from astropy.io import fits
from collections import OrderedDict
from matplotlib import animation
from ctawave.plot import TransientPlotter
from ctawave.denoise import thresholding_3d
from ctawave.toy_models_crab import simulate_steady_source_with_transient, remove_steady_background
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
    '--cu_flare',
    '-cu',
    help='Transient brightness in crab units',
    default=1.0
)
@click.option(
    '--n_bg_slices',
    '-n_bg',
    help='Number of slices for background mean',
    default=5
)
@click.option(
    '--gap',
    '-g',
    help='Minimal distance to sliding background window (number of slices). Will be enlarged by max. 3 slices if the number of slices for the resulting cube can not be divided by 4.',
    default=5
)
@click.option(
    '--cmap',
    '-c',
    help='Colormap to use for histograms',
    default='viridis'
)
def main(
            out_file,
            cu_flare,
            time_per_slice,
            time_steps,
            time_steps_bg,
            n_bg_slices,
            gap,
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

    transient_template = np.loadtxt('/home/lena/Dokumente/CTA/transient_data_2.txt')

    # enlarge gap so that the input cube for wavelet trafo has a number of slices which can be divided by 4 to perform a two level wavelet trafo.
    if((time_steps - time_steps_bg - gap) % 4 != 0):
        gap = gap + (time_steps - time_steps_bg - gap) % 4

    # create cube for steady source with transient
    cube_with_transient, trans_scale = simulate_steady_source_with_transient(
        6 * u.deg,
        6 * u.deg,
        2 * u.deg,
        2 * u.deg,
        a_eff_cta_south,
        data_bg_rate,
        ang_res_cta_south,
        cu_flare,
        transient_template,
        num_slices=time_steps,
        time_per_slice=(time_per_slice * u.s)
    )

    # remove mean measured noise from current cube
    cube = remove_steady_background(cube_with_transient, n_bg_slices, gap, [80, 80])

    # get wavelet coefficients
    coeffs = pywt.swtn(data=cube, wavelet='bior1.3', level=2, start_level=0)

    # remove noisy coefficents.
    ct = thresholding_3d(coeffs, k=30)
    cube_smoothed = pywt.iswtn(coeffs=ct, wavelet='bior1.3')
    cube_smoothed = np.concatenate(np.empty(len(cube_with_transient) - len(cube_smoothed), 80, 80), cube_smoothed)

    # some Criterion which could be used to trigger this.
    trans_factor = cube_smoothed.max(axis=1).max(axis=1)

    # hdu = fits.PrimaryHDU(cube_smoothed)
    # hdu.writeto('cube_30_100.fits')

    p = TransientPlotter(cube_with_transient,
                         cube_smoothed,
                         trans_factor,
                         trans_scale/trans_scale.max(),
                         time_per_slice,
                         cmap=cmap,
                         )

    print('Plotting animation. (Be patient)')
    anim = animation.FuncAnimation(
        p.fig,
        p.step,
        frames=(time_steps),
        interval=15,
        blit=True,
    )

    anim.save(out_file, writer='imagemagick', fps=25)
    plt.show()


if __name__ == '__main__':
    main()
