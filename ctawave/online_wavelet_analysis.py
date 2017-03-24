import numpy as np
# import astropy.units as u
# from matplotlib import animation
from ctawave.denoise import thresholding_3d
# from ctawave.plot import TransientPlotter
import pywt
from collections import deque

class Transient(object):
    window = deque([])

    bins = [80, 80]
    bin_range = [[62.5, 78.5], [-12.4, 12.4]]

    steady_cube = deque()
    current_cube = deque()

    trigger_criterion = []
    trigger_criterion_timestamps = []

    def __init__(
                self,
                window_duration,
                step=datetime.timedelta(seconds=1),
                slices_per_cube=40,
            ):
        self.slices_per_cube = slices_per_cube
        self.window_duration = window_duration
        self.step = step

    def _current_window_size(self):
        t_min, alt, az = self.window.popleft()
        self.window.appendleft((t_min, alt, az))

        t_max, alt, az = self.window.pop()
        self.window.append((t_max, alt, az))
        return t_max - t_min

    def _reduce_to_duration(self, max_duration):
        # now remove as many points we need from beginning of queue
        while self._current_window_size() > max_duration:
            self.window.popleft()

    def add_point(self, t, alt, az):

        self.window.append((t, alt, az))

        if self._current_window_size() < self.window_duration:
            return

        points = np.array(self.window)
        steady, current = np.array_split(points, 2)

        _, steady_cube = self.create_cube(steady)
        timestamps, current_cube = self.create_cube(current)

        t = self.denoise_and_compare_cubes(steady_cube, current_cube)
        self.trigger_criterion.append(list(t))
        self.trigger_criterion_timestamps.append(list(timestamps))

        has_triggered = self.check_trigger(t)
        new_duration = self.window_duration - self.step
        self._reduce_to_duration(new_duration)

    def check_trigger(self, t):
        return len(t[t > 80]) > 10

    def create_cube(self, points):
        t, alt, az = points.T

        alt = alt.astype(np.float)
        az = az.astype(np.float)

        _, x_edges, y_edges = np.histogram2d(
                    alt,
                    az,
                    bins=self.bins,
                    range=self.bin_range
        )

        slices = []
        timestamps = []
        for indeces in np.array_split(np.arange(0, len(points)), self.slices_per_cube):
            timestamps.append(t[indeces][0])
            H, _, _ = np.histogram2d(
                            alt[indeces],
                            az[indeces],
                            bins=[x_edges, y_edges],
                            range=self.bin_range)
            slices.append(H)

        slices = np.array(slices)
        timestamps = np.array(timestamps)

        return timestamps, slices

    def denoise_and_compare_cubes(self, steady_cube, cube_with_transient):
        cube = cube_with_transient - steady_cube.mean(axis=0)
        coeffs = pywt.swtn(data=cube, wavelet='bior1.3', level=2,)

        # remove noisy coefficents.
        ct = thresholding_3d(coeffs, k=30)
        cube_smoothed = pywt.iswtn(coeffs=ct, wavelet='bior1.3')

        # some Criterion which could be used to trigger this.
        trans_factor = cube_smoothed.max(axis=1).max(axis=1)

        # return trans_factor

        # p = TransientPlotter(cube_with_transient,
        #                      cube_smoothed,
        #                      trans_factor,
        #                      cmap='viridis',
        #                      )
        #
        # print('Plotting animation. (Be patient)')
        # anim = animation.FuncAnimation(
        #     p.fig,
        #     p.step,
        #     frames=len(cube),
        #     interval=15,
        #     blit=True,
        # )
        #
        # anim.save('build/anim_{}.gif'.format(self.window.popleft()[0]), writer='imagemagick', fps=25)

        return trans_factor
