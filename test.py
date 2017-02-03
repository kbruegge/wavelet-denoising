import numpy as np
import matplotlib.pyplot as plt

import pyfits

import pywt
import pywt.data

noise_sigma = 2
wavelet = 'bior6.8'

# Load image
# original_image = pywt.data.camera()
hdu_list = pyfits.open('./run1001.simtel.gz_TEL001_EV00507.fits')
# get adc sums and take the second gain channel
original_image = hdu_list[2].data[1]

noise = np.random.normal(loc=0, scale=noise_sigma, size=original_image.shape)
noised_image = original_image + noise

#
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
# ax1.imshow(original_image, interpolation='nearest', cmap='gray')
# ax2.imshow(noised_image, interpolation='nearest', cmap='gray')

# original_image = noised_image


# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

level = pywt.swt_max_level(len(original_image))

# level = 4
print('maximum level of decomposition: {}'.format(level))

coeff_list = pywt.swt2(original_image, wavelet, level)

fig, axs = plt.subplots(nrows=level, ncols=4)
for i, coeffs in enumerate(coeff_list):
    cA, (cH, cV, cD) = coeffs
    axes_row = axs[i]
    for ax, c in zip(axes_row, [cA, cH, cV, cD]):
        ax.imshow(c, origin='image', interpolation="nearest", cmap=plt.cm.gray)

fig.suptitle("swt2 coefficients", fontsize=12)


# at this points we have al coefficients for all planes. We can de-noise by adapting
# small coefficients. Some call this step 'thresholding'.
# When small, or better yet insignificant,
# coefficents are set to 0 without touching the other coefficients
# the process is called hard thresholding.

# Now assume a fixed sigma for the input data noise
def denoise(coefficient_list, sigma_d=2, k=3, kind='hard',
            sigma_levels=[0.889, 0.2, 0.086, 0.041, 0.020, 0.010, 0.005, 0.0025, 0.0012]):
    r = []
    for level, coeffs in enumerate(coefficient_list):
        cA, cs = coeffs
        cs = tuple([pywt.threshold(c, sigma_d*k*sigma_levels[level], kind) for c in cs])
        r.append((cA, cs))
    return r

cmap = 'viridis'
# Now reconstruct and plot the original image
reconstructed_image = pywt.iswt2(denoise(coeff_list, sigma_d=noise_sigma, kind='hard'), wavelet)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
im = ax1.imshow(original_image, interpolation='nearest', cmap=cmap)
fig.colorbar(im, ax=ax1)
ax1.set_title('original')
im = ax2.imshow(reconstructed_image, interpolation='nearest', cmap=cmap)
fig.colorbar(im, ax=ax2)
ax2.set_title('reconstructed')


im = ax3.imshow(original_image - reconstructed_image, interpolation='nearest', cmap=cmap)
fig.colorbar(im, ax=ax3)
ax3.set_title('residual')

# # Check that reconstructed image is close to the original
# np.testing.assert_allclose(original, reconstructed, atol=1e-3, rtol=1e-3)

plt.show()

# _b3spline1d = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
# __x = _b3spline1d.reshape(1,-1)
# _b3spl2d = np.dot(__x.T,__x)
#
# def atrous2d(arr, lev, kernel=_b3spl2d, boundary='symm'):
#     "Do 2d a'trous wavelet transform with B3-spline scaling function"
#     approx = signal.convolve2d(arr, kernel,
#                                mode='same',
#                                boundary=boundary)  # approximation
#     w = arr - approx                               # wavelet details
#     if lev <= 0: return arr
#     if lev == 1: return [w, approx]
#     else:        return [w] + atrous1(approx,lev-1, upscale(kernel), boundary)
