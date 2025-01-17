import numpy as np
from scipy import fft
from scipy import signal as sg

def zero_pad(image, shape, position='corner'):
    """Changes shape of the given array to the specified shape filling the extra points with zeros."""
    shape = np.asarray(shape, dtype=int) #converts given shape int to numpy array 
    imshape = np.asarray(image.shape, dtype=int) #extracts shape of the image and stores it as numpy array

    if np.all(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image   # zero padding step
    return pad_img


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Also makes it equal to parameter shape, given as image size, through zero padding
    """
    psf_padded = np.zeros(shape) 
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf #zero pads the given psf
    psf_padded = np.roll(psf_padded, -psf.shape[0] // 2, axis=0)  # moves image to proper axis horizontally
    psf_padded = np.roll(psf_padded, -psf.shape[1] // 2, axis=1)  # moves image to proper axis vertically
    return np.fft.fft2(psf_padded) # fourier transforms the psf to convert it to otf

def deconvwnr(image,psf,nr):
    """Performs Wiener deconvolution with the specified gamma - noise level"""
    H = psf2otf(psf, image.shape)
    denom = np.abs(H) ** 2 + nr 
    G = np.conj(H) / denom
    # Apply the Wiener filter in the frequency domain
    J = (np.abs(fft.ifft2(G * fft.fft2(image))))**2
    return J
    
def deconvwnrwindow(image,psf,nr):
    """Performs Wiener deconvolution as well as applies triangular apodisation with the specified gamma - noise level"""
    H = psf2otf(psf, image.shape)
    denom = np.abs(H) ** 2 + nr
    G = np.conj(H) / denom
    # Apply the Wiener filter in the frequency domain and perform apodisation
    J = (np.abs(fft.ifft2(G * fft.fft2(image) * window)))**2
    return J
