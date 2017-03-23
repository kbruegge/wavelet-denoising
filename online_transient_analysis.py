# import click
import dateutil.parser
import datetime
import matplotlib.pyplot as plt
import numpy as np
# import astropy.units as u
from matplotlib import animation
from ctawave.denoise import thresholding_3d
from ctawave.plot import TransientPlotter
import pywt
from collections import deque
import random
from tqdm import tqdm
plt.style.use('ggplot')


class Transient(object):
    window = deque([])
    # steady_cube = deque([])
    bins = [80, 80]
    bin_range = [[62.5, 78.5], [-12.4, 12.4]]

    steady_cube = deque()
    current_cube = deque()

    cube_counter = 0
    t_0 = None

    def __init__(self, slices_per_cube, step_size=0.2):
        self.slices_per_cube = slices_per_cube
        self.step_size = step_size

    def add_point(self, t, alt, az):
        if not self.t_0:
            self.t_0 = t

        self.window.append((alt, az))

        if t - self.t_0 < datetime.timedelta(seconds=10):
            return

        print(len(self.window))
        self.t_0 = t
        steady, current = np.array_split(np.array(self.window), 2)
        steady_cube = self.create_cube(steady[:, 0], steady[:, 1])
        current_cube = self.create_cube(current[:, 0], current[:, 1])

        self.cube_counter += 1
        t = self.denoise_and_compare_cubes(steady_cube, current_cube)

        # now remove as many points we need from beginning of queue
        for _ in range(int(self.step_size*len(self.window))):
            self.window.popleft()

        has_triggered = self.check_trigger(t)
        if has_triggered:
            print('OMG a trigger')

    def check_trigger(self, t):
        return len(t[t > 20]) > 10

    def create_cube(self, alt, az):
        _, x_edges, y_edges = np.histogram2d(
                    alt,
                    az,
                    bins=self.bins,
                    range=self.bin_range
        )

        N = len(alt)

        slices = []
        for indeces in np.array_split(np.arange(0, N), self.slices_per_cube):
            H, _, _ = np.histogram2d(
                            alt[indeces],
                            az[indeces],
                            bins=[x_edges, y_edges],
                            range=self.bin_range)
            slices.append(H)

        slices = np.array(slices)

        return slices

    def denoise_and_compare_cubes(self, steady_cube, cube_with_transient):
        cube = cube_with_transient - steady_cube.mean(axis=0)
        coeffs = pywt.swtn(data=cube, wavelet='bior1.3', level=2,)

        # remove noisy coefficents.
        ct = thresholding_3d(coeffs, k=30)
        cube_smoothed = pywt.iswtn(coeffs=ct, wavelet='bior1.3')

        # some Criterion which could be used to trigger this.
        trans_factor = cube_smoothed.max(axis=1).max(axis=1)

        # return trans_factor

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

        anim.save('anim{}.gif'.format(self.cube_counter), writer='imagemagick', fps=25)

        return trans_factor

# @click.command()
# @click.argument('data_file', type=click.Path(exists=True))
# @click.argument('output_file', type=click.Path(exists=False))
def main():


    target_region = [[69.5, 70.5], [-0.5, 0.5]]
    transient = Transient(40, step_size=0.4)
    alt_range = np.array(transient.bin_range[0])
    az_range = np.array(transient.bin_range[1])

    t_background = datetime.datetime.utcnow()
    t_signal = t_background

    for i in tqdm(range(15*10000)):

        alt = random.uniform(*alt_range)
        az = random.uniform(*az_range)
        t_background = t_background + datetime.timedelta(seconds=0.0001)

        transient.add_point(t_background, alt, az)

    for i in tqdm(range(20*10000)):

        alt = random.uniform(*alt_range)
        az = random.uniform(*az_range)
        t_background = t_background + datetime.timedelta(seconds=0.0001)

        transient.add_point(t_background, alt, az)

        alt = random.uniform(*target_region[0])
        az = random.uniform(*target_region[1])
        t_signal = t_background + datetime.timedelta(seconds=random.uniform(0.00001, 0.0001))
        transient.add_point(t_signal, alt, az)

    for i in tqdm(range(25*10000)):

        alt = random.uniform(*alt_range)
        az = random.uniform(*az_range)
        t_background = t_background + datetime.timedelta(seconds=0.0001)

        transient.add_point(t_background, alt, az)

    #
    # for i in tqdm(range(5000*1*100)):
    #     alt = random.uniform(*alt_range)
    #     az = random.uniform(*az_range)
    #     t += 0.0001
    #     transient.add_point(t, alt, az)
    #
    #     alt = random.uniform(*(alt_range*0.05))
    #     az = random.uniform(*(az_range*0.05))
    #     t += 0.00005
    #     transient.add_point(t, alt, az)
    #
    # for i in tqdm(range(5000*1*100)):
    #
    #     alt = random.uniform(*alt_range)
    #     az = random.uniform(*az_range)
    #     t += 0.0001
    #
    #     transient.add_point(t, alt, az)

if __name__ == "__main__":
    main()
