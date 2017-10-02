import matplotlib.pyplot as plt
import pywt
import click
import numpy as np


from astropy.io import fits

from matplotlib import animation
from IPython import embed
from ctawave.plot import TransientPlotter
from ctawave.denoise import thresholding_3d
from ctawave.toy_models_crab import remove_steady_background
plt.style.use('ggplot')


@click.command()
@click.argument('input_file', type=click.Path(file_okay=True, dir_okay=False))
@click.argument('out_file', type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    '--time_per_slice',
    '-t',
    type=click.INT,
    help='Measuring time for one slice in s',
    default='30'
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
@click.option(
    '--bins',
    '-b',
    help='Colormap to use for histograms',
    default=80
)


def main(
    input_file,
    out_file,
    time_per_slice,
    n_bg_slices,
    gap,
    cmap,
    bins
        ):

        out_file_cube_raw = 'cube_raw.fits'
        out_file_cube_smoothed = 'cube_smoothed.fits'
        outfile_trans_factor = 'trans_factor.fits'

        cube_raw = fits.open(input_file)[0].data.reshape([-1, bins, bins])
        if((cube_raw.shape[0] - n_bg_slices - gap) % 4 != 0):
            gap = gap + (cube_raw.shape[0] - n_bg_slices - gap) % 4
        cube = remove_steady_background(cube_raw, n_bg_slices, gap, [bins, bins])

        # get wavelet coefficients

        coeffs = pywt.swtn(data=cube, wavelet='bior1.3', level=2, start_level=0)

        # remove noisy coefficents.
        ct = thresholding_3d(coeffs, k=30)
        cube_smoothed = pywt.iswtn(coeffs=ct, wavelet='bior1.3')
        # embed()
        cube_smoothed = np.concatenate([np.zeros([len(cube_raw) - len(cube_smoothed), bins, bins]), cube_smoothed])

        # some Criterion which could be used to trigger this.
        trans_factor = cube_smoothed.max(axis=1).max(axis=1)

        # diff_trans_factor = np.diff(trans_factor)

        # embed()

        hdu_cube_raw = fits.PrimaryHDU(cube_raw)
        hdu_cube_smoothed = fits.PrimaryHDU(cube_smoothed)
        hdu_trans_factor = fits.PrimaryHDU(trans_factor)
        hdu_cube_raw.writeto(out_file_cube_raw, overwrite=True)
        hdu_cube_smoothed.writeto(out_file_cube_smoothed, overwrite=True)
        hdu_trans_factor.writeto(outfile_trans_factor, overwrite=True)




        # p = TransientPlotter(cube_raw,
        #                      cube_smoothed,
        #                      trans_factor,
        #                      trans_factor,
        #                      time_per_slice,
        #                      cmap=cmap,
        #                      )
        #
        # print('Plotting animation. (Be patient)')
        # anim = animation.FuncAnimation(
        #     p.fig,
        #     p.step,
        #     frames=(len(cube_smoothed)),
        #     interval=50,
        #     blit=True,
        # )
        #
        # anim.save(out_file, writer='imagemagick', fps=25)
        # plt.show()


if __name__ == '__main__':
    main()
