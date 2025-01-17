import tifffile
import matplotlib.pyplot as plt

#reads and plots the widefield image output of SOFI simulation tool
WF = tifffile.imread("raw_files/widefield.tif")
plt.imshow(WF, cmap='hot')
plt.axis("Off")
plt.show()
#plt.imsave('results/widefield.png',WF,cmap='hot')
