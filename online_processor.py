import Pyro4
# import numpy as np
from datetime import timedelta
import astropy.units as u
from dateutil.parser import parse
from astropy.coordinates import cartesian_to_spherical, Angle
from ctawave.online_wavelet_analysis import Transient
from ctawave.plot import TransientPlotter


@Pyro4.expose
class Processor(object):
    transient = Transient(
                        window_duration=timedelta(seconds=1),
                        step=timedelta(seconds=0.4),
                        bins=[12, 12],
                        slices_per_cube=12
                    )

    item_counter = 0

    def process(self, item):

        alt, az = self.altAz(item)
        t = parse(item['timestamp'])
        self.transient.add_point(t, alt, az)

        self.item_counter += 1
        if self.item_counter % 1024 == 0:
            print('Current window length: {}'.format(self.transient._current_window_size()))

            if self.transient.trigger_criterion:
                fig = TransientPlotter.plot_trigger_criterion(self.transient)
                fig.savefig('trans_{}.png'.format(self.item_counter))

        return item

    def altAz(self, df):

        x = df['stereo:estimated_direction:x']
        y = df['stereo:estimated_direction:y']
        z = df['stereo:estimated_direction:z']

        _, lat, lon = cartesian_to_spherical(
            x * u.m, y * u.m, z * u.m)

        alt = Angle(90 * u.deg - lat).degree
        az = Angle(lon).wrap_at(180 * u.deg).degree
        return alt, az


def main():
    Pyro4.Daemon.serveSimple(
            {
                Processor: 'streams.processors'
            },
            ns=True
    )

if __name__ == '__main__':
    main()
