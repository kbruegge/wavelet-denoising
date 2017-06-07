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
def main(
            out_file,
            time_per_slice,
            time_steps,
            cmap
        ):
    '''
    Use a crab toy model to create a transient appearing in the FoV of another source.
    A steady background is subtracted and denoised using wavelets.
    This script then creates an animated gif of the whole shebang saved under the
    OUT_FILE argument.
    '''

    cta_perf_fits = fits.open('/home/lena/Dokumente/CTA/prod3b-caldb-20170502/caldb/data/cta/prod3b/bcf/South_z20_100s/irf_file.fits')
    data_A_eff = cta_perf_fits[1]
    data_ang_res = cta_perf_fits[2]
    data_bg_rate = cta_perf_fits[4]

    a_eff_cta_south = pd.DataFrame(OrderedDict({"E_TeV": (data_A_eff.data['ENERG_LO'][0] + data_A_eff.data['ENERG_HI'][0])/2, "A_eff": data_A_eff.data['EFFAREA'][0].mean(axis=0)}))
    ang_res_cta_south = pd.DataFrame(OrderedDict({"E_TeV": (data_ang_res.data['ENERG_LO'][0] + data_ang_res.data['ENERG_HI'][0])/2, "Ang_Res": data_ang_res.data['SIGMA'][0][0]}))
    bg_rate_south = pd.DataFrame(OrderedDict({"E_TeV": (data_bg_rate.data['ENERG_LO'][0] + data_bg_rate.data['ENERG_HI'][0])/2, "bg_rate": data_bg_rate.data['BGD'][0].sum(axis=1).sum(axis=1)}))

    # create cubes for steady source and steady source with transient
    cube_steady = simulate_steady_source(6 * u.deg, 6 * u.deg, a_eff_cta_south, bg_rate_south, ang_res_cta_south, num_slices=time_steps, time_per_slice=time_per_slice * u.s)
    cube_with_transient = simulate_steady_source_with_transient(6 * u.deg, 6 * u.deg, 2 * u.deg, 2 * u.deg, a_eff_cta_south, bg_rate_south, ang_res_cta_south, num_slices=time_steps, time_per_slice=time_per_slice * u.s)

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
