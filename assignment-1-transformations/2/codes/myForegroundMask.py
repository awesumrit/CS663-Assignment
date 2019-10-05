import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def threshold_img(img_path, k=50):
    img = Image.open(img_path)
    img = np.array(img)
    mask = 255 * (img > k)
    masked_img = img * (img > k)

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Foreground Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(masked_img, cmap='gray')
    plt.title('Masked Image')
    plt.show()

    return (img > k).astype(np.int)

if __name__ == "__main__":
    threshold_img('../data/statue.png')
