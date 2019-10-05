from matplotlib import pyplot as plt

from myForegroundMask import threshold_img
from myLinearContrastStretching import linearContrastStretch
from myHE import histogramEqualize
from myHM import histogramMatch
from myCLAHENew import clahe

images = {
            1: '../data/barbara.png',
            2: '../data/TEM.png',
            3: '../data/canyon.png',
            4: '../data/retina.png',
            5: '../data/church.png',
            6: '../data/chestXray.png',
            7: '../data/statue.png',
            8: '../data/retinaRef.png',
            9: '../data/retinaMask.png',
            10: '../data/retinaRefMask.png',
        }

# Set figure width to 12 and height to 9
fig_size = [15, 12]
plt.rcParams["figure.figsize"] = fig_size

# Foreground Mask
mask = threshold_img(images[7], k=60)

# Linear Contrast stretching
for i in [1, 2, 3, 5, 6]:
    linearContrastStretch(images[i])
linearContrastStretch(images[7], mask=mask)

# Histogram Equalization
for i in [1, 2, 3, 5, 6]:
    histogramEqualize(images[i])
histogramEqualize(images[7], mask)

# Histogram Matching
histogramMatch(images[4], images[8], images[9], images[10])

# Contrast-Limited Adaptive Histogram Equalization
for i in [1,2,3,6]:
    for window_size in [30, 60, 120]:
            clahe(images[i], window_size, 0.05)

for i in [1,2,3,6]:
    clahe(images[i], window_size, 0.025)
