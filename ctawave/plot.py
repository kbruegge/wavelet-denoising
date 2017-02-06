import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np


def pixel_histogram(*images, labels=['reconstructed pixel values'], bins=40):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for image, label in zip(images, labels):
        ax.hist(image.flatten(), bins=bins, histtype='step', label=label)

    plt.legend()
    return fig

def coefficients(coeff_list, cmap='gray'):
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    fig, axs = plt.subplots(nrows=len(coeff_list), ncols=4)
    for i, coeffs in enumerate(coeff_list):
        cA, (cH, cV, cD) = coeffs
        axes_row = axs[i]
        for ax, c in zip(axes_row, [cA, cH, cV, cD]):
            im = ax.imshow(c, origin='image', interpolation="nearest", cmap=cmap)
            ax.tick_params(labelbottom='off', labelleft='off')
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=6)
            # ax.tick_params(axis='both', which='both', labelsize=4)

    fig.suptitle("swt2 coefficients", fontsize=12)
    fig.tight_layout()

    return fig

def results(original_image, reconstructed_image, npe_truth, cmap='viridis'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    im = ax1.imshow(original_image, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax1)
    ax1.set_title('original')
    ax1.tick_params(labelbottom='off', labelleft='off')

    im = ax2.imshow(reconstructed_image, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax2)
    ax2.set_title('reconstructed')
    ax2.tick_params(labelbottom='off', labelleft='off')

    im = ax3.imshow(original_image - reconstructed_image, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax3)
    ax3.set_title('residual')
    ax3.tick_params(labelbottom='off', labelleft='off')

    im = ax4.imshow(npe_truth, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax4)
    ax4.set_title('simulated photons')
    ax4.tick_params(labelbottom='off', labelleft='off')

    return fig
