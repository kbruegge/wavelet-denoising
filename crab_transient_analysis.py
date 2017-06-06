import matplotlib.pyplot as plt
import pandas as pd
import pywt
import click

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

    ang_res_cta_south = pd.read_csv(
        '/home/lena/Dokumente/CTA/CTA-Performance-South-20150511.ASCII1_/CTA-Performance-South-Angres.txt',
        names=['E_TeV', 'Ang_Res'],
        skiprows=10,
        sep='\s+',
        index_col=False
        )

    a_eff_cta_south = pd.read_csv(
        '/home/lena/Dokumente/CTA/CTA-Performance-South-20150511.ASCII1_/CTA-Performance-South-0.5h-EffArea.txt',
        names=['E_TeV', 'A_eff'],
        skiprows=10,
        sep='\s+',
        index_col=False)

    cta_perf_fits = fits.open('/home/lena/Dokumente/CTA/prod3b-caldb-20170502/caldb/data/cta/prod3b/bcf/South_z20_100s/irf_file.fits')
    data_A_eff = cta_perf_fits[1]
    data_bg_rate = cta_perf_fits[4]
    # area = h.data['EFFAREA']
    # a = area[0].mean(axis=0)
    # x = h.data['ENERG_LO'][0]
    # a_eff_cta_south_bg = pd.DataFrame(OrderedDict({"E": data_A_eff.data['ENERG_LO'][0], "A_eff": data_A_eff.data['EFFAREA'][0].mean(axis=0)}))
    bg_rate_south = pd.DataFrame(OrderedDict({"E_low": data_bg_rate.data['ENERG_LO'][0], "E_high": data_bg_rate.data['ENERG_HI'][0], "bg_rate": data_bg_rate.data['BGD'][0].sum(axis=1).sum(axis=1)}))

    cube_steady = simulate_steady_source(6, 6, a_eff_cta_south, bg_rate_south, ang_res_cta_south, num_slices=time_steps, time_per_slice=time_per_slice)
    cube_with_transient = simulate_steady_source_with_transient(6, 6, 2, 2, a_eff_cta_south, bg_rate_south, ang_res_cta_south, num_slices=time_steps, time_per_slice=time_per_slice)

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
