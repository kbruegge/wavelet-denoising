#Wavelet Denoising

Testing some methods to denoise different types of images using a stationary wavelet
transform. So far this contains only scripts for testing and visual inspection.


#Notes

### Wavelets and stuff

So the whole motiviation for this is to find out whats going on exactly in
the programm called mr_filter provided by the c++ library written by the
cosmostat group.

mr_filter uses, allegedly,  a bspline wavelet transform (if you supply no other option)

The coefficients stored in the code are.

    float Coeff_h0 = 3. / 8.;
    float Coeff_h1 = 1. / 4.;
    float Coeff_h2 = 1. / 16.;

which is crazy speak for something symmetric I presume.
While i'm not quite sure yet how these coefficients connect to those
used during the construction of Splines or Bspline Basis functions I presume
the coefficients c = [1/16, 1/4, 3/8 , 1/4, 1/16] make more sense.
Interestingly this corresponds to

    1/2 * (coefficients of the two scale relationship of the cubic bspline)

The outer product of the c with itself yields this matrix.

    In [18]: np.outer(c, c)
    Out[18]:
    array([[ 0.00390625,  0.015625  ,  0.0234375 ,  0.015625  ,  0.00390625],
       [ 0.015625  ,  0.0625    ,  0.09375   ,  0.0625    ,  0.015625  ],
       [ 0.0234375 ,  0.09375   ,  0.140625  ,  0.09375   ,  0.0234375 ],
       [ 0.015625  ,  0.0625    ,  0.09375   ,  0.0625    ,  0.015625  ],
       [ 0.00390625,  0.015625  ,  0.0234375 ,  0.015625  ,  0.00390625]])

This is the convolution kernel used during the actual transformation of the image.

Now lets try and translate that to a more commonly used name for wavelets.

First important thing to notice is, that there are two common names for the same thing.

  1. Father Wavelet and Scaling function. These two terms have the same meaning.

  2. Also spline-wavelet seems to be a common term for the special class of wavelets constructed
  from BSplines. These are not orthogonal but they do have compact support. which is nice.

  3. According to [2 p. 154, 270 ff.] a biorthogonal discrete wavelet transformation (without subsampling)
  is similar to the UWT.

To produce a wavelet within the pywt library we'd need the coefficents of the scaling filter.
I don't how to get them yet. However the default scaling function used in mr_filter is
called `biorthogonal 7/9`. Pywt and matlab provide biorthogonals up tp 6 and 8.


Anyhow here goes thresholding.
So according to the help text of mr_filter the call thats ususally made to the binary
, "-K -C1 -m3 -s3 -n4",  translates to:

  1. Perform k-sigma clipping
  2. Noise type is gaussian plus poisson. (this assumes > 20 counts in each pixel).
  3. The K option suppresses the last scale. Why this should improve things is unclear to me
  that is unclear to me.

The sigma clipping part estimates the noise in the image. This might not be necessary
since we know, at least parts of, the image noise from pedestal measurements.

Assuming Gaussian Noise in the wavelet coefficients, and in turn also in the original image,
allows for simple hypothesis testing. A mixture of Gaussian and Poisson noise is treated by
changing the image values with help of the Anscombe-Transformation. At least according to [1].

We can de-noise by adapting or removing small wavelet coefficients. This process is called 'thresholding'.
When small, or better yet insignificant, coefficents are set to 0 without touching the other coefficients
the process is called hard thresholding.
In this code I simply apply hard thresholding in each level of the transformed image. The default
threshold values are copied from [1] but they should be adapted for different Wavelet/Scalings.
So right now these thresholds can be considered arbritary.




[1] J.Starck. Astronomical Image and Data Analysis.
[2] S.Mallat A wavelet tour of signal processing.
