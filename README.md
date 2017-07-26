# Sliding Window

Noise is reduced by subtraction of the mean over a sliding window from every slice.
The window has a width of 5 slices per default. There is a gap of a couple of slices between the current slice and the window.
The width of these gap depends on the number of simulated timesteps(slices). Since the number of
slices that can be processed by a 2-level wavelet transformation has to be divisble by 4, the gap has a default of 5 slices
and is enlarged up to 8 in case the number of slices of the resulting cube does not fit for the wavelet transformation.


The following gif is a simulation for a transient with a maximal brightness of one crab unit and crab as steady source.
Every slice contains an observation time of 30 seconds.
Background rates and angular resolution are based on the instrument response functions of CTA (Release 2017-06-27).

![wavelet](https://raw.githubusercontent.com/mackaiver/wavelet-denoising/sliding_bg_window/transient_sw.gif)
