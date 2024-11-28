import numpy as np
import matplotlib.pyplot as plt
import tifffile
from deconvwnr import *
from scipy import fft

y = tifffile.imread('raw_files/reconstruct_icm.tif')
psf = tifffile.imread('raw_files/PSF RW.tif')
WF = tifffile.imread('raw_files/widefield.tif')

for i in range(25):
    y[i] /= y[i].max()

psf = psf[8,:,:]
psf /= psf.max()

window1d = np.abs(sg.windows.triang(64))
window = np.sqrt(np.outer(window1d,window1d))

for i in range(25):
    plt.imsave(f'raw_files/frame{i}.png',y[i],cmap='hot')

plt.imsave('raw_files/psf.png',psf,cmap='hot')

deconvolved = np.zeros((64,64))
psf = plt.imread('raw_files/psf.png')[:,:,0]

for i in range(25):
    dec = deconvwnr(plt.imread(f'raw_files/frame{i}.png')[:,:,0],psf,0.05)
    dec /= dec.max()
    deconvolved += dec

deconvolved /= deconvolved.max()
windowed = (np.abs(fft.ifft2(fft.fft2(deconvolved) * window)))**2

fig = plt.figure(figsize = (14,7))
fig.add_subplot(1,2,1)
plt.imshow(WF,cmap='hot')
plt.axis("Off")
fig.add_subplot(1,2,2)
plt.imshow(windowed,cmap='hot')
plt.tight_layout()
plt.axis("Off")
plt.imsave("results/SIM_ICM.png",windowed,cmap='hot')
plt.show()
