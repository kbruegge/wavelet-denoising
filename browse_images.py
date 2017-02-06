from ctawave import io, plot, denoise
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt
import pywt.data

from matplotlib.backends.backend_pdf import PdfPages


def main():
    noise_sigma = 2
    wavelet = 'bior1.3'

    images = list(io.load_fits_images_in_folder('./resources'))
    with PdfPages('multipage_pdf.pdf') as pdf:
        for i, (calibrated_image, npe_truth) in tqdm(enumerate(images)):

            fig = plt.figure()
            plt.text(1, 1, 'Event Number: {}'.format(i))
            plt.xlim(0, 2)
            plt.ylim(0, 2)
            plt.axis('off')
            pdf.savefig(fig)
            plt.close()

            # transform the image
            level = pywt.swt_max_level(len(calibrated_image))
            # print('maximum level of decomposition: {}'.format(level))

            coeff_list = pywt.swt2(calibrated_image, wavelet, level)

            fig = plot.coefficients(coeff_list)
            pdf.savefig(figure=fig)
            plt.close()

            levels = [0.889, 0.7, 0.586]
            coeff_list = denoise.thresholding(coeff_list,
                                              sigma_d=noise_sigma,
                                              kind='hard',
                                              sigma_levels=levels)

            reconstructed_image = pywt.iswt2(coeff_list, wavelet)

            fig = plot.results(calibrated_image, reconstructed_image, npe_truth)
            pdf.savefig(figure=fig)
            plt.close()

            fig = plot.pixel_histogram(
                reconstructed_image,
                calibrated_image,
                npe_truth, labels=['reconstructed', 'calibrated', 'npe mc truth'], bins=60)

            pdf.savefig(figure=fig)
            plt.close()


if __name__ == '__main__':
    main()
