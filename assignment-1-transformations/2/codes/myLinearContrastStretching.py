import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def linearContrastStretch(img_path, mask=None):
    img = Image.open(img_path)
    img = np.array(img)

    if mask is not None:
        img = mask * img + (1 - mask)
        img_m = img.reshape(-1)
        img_m = img_m[np.where(img_m>=0)]
    else:
        img_m = img.reshape(img.shape[0]*img.shape[1], -1)


    img_min, img_max = img_m.min(0), img_m.max(0)
    curr_range = img_max - img_min
    img_streched = ((img - img_min)/curr_range)*255
    img_streched = img.astype(np.uint8)

    a,b = 1, 2
    if img.shape[0] > img.shape[1]*2:
        a,b = 2,1

    plt.subplot(a, b, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    if img.ndim == 2:
        plt.colorbar()

    plt.subplot(a, b, 2)
    plt.imshow(img_streched, cmap='gray')
    plt.title('Contrast Stretched Image')
    if img.ndim == 2:
        plt.colorbar()
    plt.show()


if __name__ == "__main__":
    linearContrastStretch('../data/church.png')
