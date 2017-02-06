import os
import pyfits


def load_fits_images_in_folder(path='./'):
    index = {'adc sum images': 2,
             'calibrated image': 0,
             'calibration images': 5,
             'gains images': 4,
             'pe image': 1,
             'pedestal images': 3,
             'pixels position': 6}
    # Load image
    # original_image = pywt.data.camera()
    for f in os.listdir(path):
        if f.endswith('.fits'):
            hdu_list = pyfits.open(os.path.join(path, f))
            # get adc sums and take the second gain channel
            calibrated_image = hdu_list[index['calibrated image']].data
            npe_truth = hdu_list[index['pe image']].data
            yield (calibrated_image, npe_truth)
