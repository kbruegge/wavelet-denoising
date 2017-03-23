import click
import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical, Angle
from ctawave.plot import CubePlotter, TransientPlotter
from ctawave.denoise import thresholding_3d
import pywt
plt.style.use('ggplot')


def loadFile(input_file):
    df = pd.read_csv(input_file).dropna()

    x = df['stereo:estimated_direction:x']
    y = df['stereo:estimated_direction:y']
    z = df['stereo:estimated_direction:z']

    _, lat, lon = cartesian_to_spherical(
        x.values * u.m, y.values * u.m, z.values * u.m)

    alt = Angle(90 * u.deg - lat).degree
    az = Angle(lon).wrap_at(180 * u.deg).degree
    df['alt'] = alt
    df['az'] = az
    return df


def create_cube(df, bins, bin_range):
    _, x_edges, y_edges = np.histogram2d(
        df.alt, df.az, bins=bins, range=bin_range)
    slices = []
    N = 100
    for df in np.array_split(df, N):
        H, _, _ = np.histogram2d(df.alt, df.az, bins=[
                                 x_edges, y_edges], range=bin_range)
        slices.append(H)

    slices = np.array(slices)
    return slices


@click.command()
@click.argument('gamma_file', type=click.Path(exists=True))
@click.argument('proton_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
def main(gamma_file, proton_file, output_file):
    bins = [80, 80]
    bin_range = [[62.5, 78.5], [-12.4, 12.4]]

    df_gammas = loadFile(gamma_file)
    df_protons = loadFile(proton_file)

    print('Read {} gammas and {} protons'.format(len(df_gammas), len(df_protons)))
    factor = (10E5 * len(df_gammas)) / len(df_protons)

    print(factor)

    df_background = df_protons[df_protons['prediction:signal:mean'] > 0.87]
    df_signal = df_gammas[df_gammas['prediction:signal:mean'] > 0.87]


    print('Read {} signal events and {} background events'.format(len(df_signal), len(df_background)))
    ratio = len(df_background)/len(df_signal)
    expected_background = int(ratio * len(df_background) * factor)
    print('Upsampling background to get {} events'.format(expected_background))
    df_background = df_protons.sample(expected_background, replace=True)


    cube_background = create_cube(
        df_background.sample(frac=0.5), bins=bins, bin_range=bin_range)
    cube_gammas = create_cube(df_gammas.sample(frac=0.5), bins=bins, bin_range=bin_range)

    cube_steady = cube_background + cube_gammas

    cube_background = create_cube(
        df_background.sample(frac=0.5), bins=bins, bin_range=bin_range)
    cube_gammas = create_cube(df_gammas.sample(frac=0.5), bins=bins, bin_range=bin_range)
    cube_bright_gammas = create_cube(df_gammas, bins=bins, bin_range=bin_range)

    cube_with_transient = np.vstack((cube_background + cube_gammas, cube_background + cube_bright_gammas))

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
                         cmap='viridis',
                         )

    print('Plotting animation. (Be patient)')
    anim = animation.FuncAnimation(
        p.fig,
        p.step,
        frames=len(cube),
        interval=15,
        blit=True,
    )

    anim.save('anim.gif', writer='imagemagick', fps=25)
    #
    #
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    #
    # (_, _, _, im) = ax1.hist2d(
    #  alt, az, range=bin_range, bins=bins, cmap='viridis',
    # )
    # ax1.set_ylabel('Azimuth')
    # ax1.set_xlim(bin_range[0])
    # ax1.set_ylim(bin_range[1])
    # ax1.get_xaxis().set_visible(False)
    # ax1.grid(b=False)
    # fig.colorbar(im, ax=ax1)
    #
    #
    #
    # (_, _, _, im) = ax2.hist2d(
    #  alt, az, range=bin_range, bins=bins,  cmap='viridis',
    # )
    # ax2.set_xlabel('Altitude')
    # ax2.set_xlim(bin_range[0])
    # ax2.set_ylim(bin_range[1])
    # ax2.grid(b=False)
    #
    # fig.colorbar(im, ax=ax2)
    #
    # # import IPython; IPython.embed()
    # # plt.savefig(output_file)
    # plt.show()


if __name__ == "__main__":
    main()
