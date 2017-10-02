import pandas as pd
import numpy as np
import click
import astropy.units as u

from collections import OrderedDict
from ctawave.toy_models_crab import simulate_steady_source_with_transient
from astropy.io import fits
from tqdm import tqdm


@click.command()
@click.argument('out_file', type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    '--n_transient',
    '-n',
    type=click.INT,
    help='Number of transients to simulate',
    default='10'
)
def main(
    out_file,
    n_transient,
    cu_min=0.1,
    cu_max=7,
    duration_min=10,
    duration_max=100
        ):
    cta_perf_fits = fits.open('/home/lena/Software/ctools-1.3.0/caldb/data/cta/prod3b/bcf/South_z20_average_100s/irf_file.fits')
    data_A_eff = cta_perf_fits['EFFECTIVE AREA']
    data_ang_res = cta_perf_fits['POINT SPREAD FUNCTION']
    data_bg_rate = cta_perf_fits['BACKGROUND']

    pks_data = np.loadtxt('/home/lena/Dokumente/CTA/transient_data_1.txt')
    hess_data = np.loadtxt('/home/lena/Dokumente/CTA/transient_data_2.txt')
    transient_templates = [pks_data, hess_data]

    a_eff_cta_south = pd.DataFrame(OrderedDict({"E_TeV": (data_A_eff.data['ENERG_LO'][0] + data_A_eff.data['ENERG_HI'][0])/2, "A_eff": data_A_eff.data['EFFAREA'][0][0]}))
    ang_res_cta_south = pd.DataFrame(OrderedDict({"E_TeV": (data_ang_res.data['ENERG_LO'][0] + data_ang_res.data['ENERG_HI'][0])/2, "Ang_Res": data_ang_res.data['SIGMA_1'][0][0]}))

    fov_min = 0 * u.deg
    fov_max = 12 * u.deg

    slices = []
    for i in tqdm(range(n_transient)):
        cube, trans_scale = simulate_steady_source_with_transient(
                    x_pos_steady_source=6*u.deg,
                    y_pos_steady_source=6*u.deg,
                    # x_pos_transient=np.random.randint(fov_min/u.deg, fov_max/u.deg)*u.deg,
                    # y_pos_transient=np.random.randint(fov_min/u.deg, fov_max/u.deg)*u.deg,
                    x_pos_transient=1*u.deg,
                    y_pos_transient=1*u.deg,
                    df_A_eff=a_eff_cta_south,
                    fits_bg_rate=data_bg_rate,
                    df_Ang_Res=ang_res_cta_south,
                    cu_flare=(cu_max - cu_min) * np.random.random() + cu_min,
                    transient_template=transient_templates[np.random.randint(len(transient_templates) - 1)],
                    num_slices=np.random.randint(duration_min, duration_max),
                    time_per_slice=30 * u.s,
                    bins=[200,200],
                    )
        slices = np.append(slices, cube)
    print('slices: {}'.format(len(slices)))

    hdu = fits.PrimaryHDU(slices)
    hdu.writeto(out_file, overwrite=True)


if __name__ == '__main__':
    main()
