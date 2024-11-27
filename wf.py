import tifffile
import matplotlib.pyplot as plt

WF = tifffile.imread("raw_files/widefield.tif")
plt.imshow(WF, cmap='hot')
plt.axis("Off")
plt.show()
#plt.imsave('results/widefield.png',WF,cmap='hot')
