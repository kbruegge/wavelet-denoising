#Wavelets and shit

mr_filter uses a bspline wavelet transform (if you supply no other option)

The coefficients stored in the code are.

    float Coeff_h0 = 3. / 8.;
    float Coeff_h1 = 1. / 4.;
    float Coeff_h2 = 1. / 16.;

which is crazy speak for something symmetric I presume.

Using the coefficients c = [1/16, 1/4, 3/8 , 1/4, 1/16] makes more sense.
This is a B3 spline in 1D. The 2D matrix is the outer product of
the c with itself.

    In [18]: np.outer(c, c)
    Out[18]:
    array([[ 0.00390625,  0.015625  ,  0.0234375 ,  0.015625  ,  0.00390625],
       [ 0.015625  ,  0.0625    ,  0.09375   ,  0.0625    ,  0.015625  ],
       [ 0.0234375 ,  0.09375   ,  0.140625  ,  0.09375   ,  0.0234375 ],
       [ 0.015625  ,  0.0625    ,  0.09375   ,  0.0625    ,  0.015625  ],
       [ 0.00390625,  0.015625  ,  0.0234375 ,  0.015625  ,  0.00390625]])

No lets try and translate that to a more commonly used name for wavelets.

First important thing to notice is, that there are two common names for the same thing.
Father Wavelet and Scaling function. These two terms have the same meaning.

To produce a wavelet within the pywt library we'd need the coefficents of the scaling filter.
I don't how to get them yet. However the default scaling function used in mr_filter is
called `biorthogonal 7/9`. Pywt and matlab provide biorthogonals up tp 6 and 8.


Anyhow here goes thresholding.
So according to the help text of mr_filter the call thats ususally made to the binary
, "-K -C1 -m3 -s3 -n4",  translates to:

 1. Perform k-sigma clipping
 2. Noise type is gaussian plus poisson. (this assumes > 20 counts).
 3. The K option suppresses the last scale. Why we do that is unclear to me.

The sigma clipping part estimates the noise in the image. This might not be necessary
since we know, at least parts of, the image noise from pedestal measurements.

Assuming Gaussian Noise in the wavelet coefficients, and therefore also in the original image,
allows for simple hypothesis testing. Assuming a mixture of Gaussian and Poisson noise can be treated by
changin the image values according to the Anscombe-Transformation. At least according to [1].




[1] J.Starck. Astronomical Image and Data Analysis.
