import tifffile
import matplotlib.pyplot as plt

WF = tifffile.imread("widefield.tif")
plt.imshow(WF, cmap='hot')
plt.axis("Off")
plt.show()
#plt.imsave('widefield.png',WF,cmap='hot')