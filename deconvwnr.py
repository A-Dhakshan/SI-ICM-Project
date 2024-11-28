import numpy as np
from scipy import fft
from scipy import signal as sg

window1d = np.abs(sg.windows.triang(64))
window = np.sqrt(np.outer(window1d,window1d))

#scipy triangular apodization function. needs to be modified to fit my values

def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

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

    pad_img[idx + offx, idy + offy] = image
    return pad_img


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Also makes it equal to parameter shape, given as image size, through zero padding
    """
    psf_padded = np.zeros(shape)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf
    psf_padded = np.roll(psf_padded, -psf.shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, -psf.shape[1] // 2, axis=1)
    return np.fft.fft2(psf_padded)

def deconvwnr(image,psf,nr):
    H = psf2otf(psf, image.shape)
    denom = np.abs(H) ** 2 + nr
    G = np.conj(H) / denom
    # Apply the Wiener filter in the frequency domain
    J = (np.abs(fft.ifft2(G * fft.fft2(image))))**2
    return J
    
def deconvwnrwindow(image,psf,nr):
    H = psf2otf(psf, image.shape)
    denom = np.abs(H) ** 2 + nr
    G = np.conj(H) / denom
    # Apply the Wiener filter in the frequency domain
    J = (np.abs(fft.ifft2(G * fft.fft2(image) * window)))**2
    return J