import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_tile(img, y, x, k):
    h, w = img.shape
    p1 = (max(0, y-k), max(0, x-k))
    p2 = (min(y+k, h-1), min(x+k, w-1))
    return p1, p2


def get_cdf(img, p1, p2, threshold=0.05):
    pdf = np.histogram(img[p1[0]:p2[0], p1[1]: p2[1]], 256, (0, 256))[0]
    pdf = pdf / ((p2[0] - p1[0]) * (p2[1] - p1[1]))
    pdf = np.clip(pdf, 0, threshold)
    pdf += (1 - np.sum(pdf))/256
    cdf = 255 * pdf.cumsum()
    cdf = cdf.astype(np.uint8)
    return cdf


def applyHE(img, p1, p2, y, x, k, w, h):
    cdf = get_cdf(img, p1, p2)
    x1, y1 = k, k
    if x <= k:
        x1 = x
    elif x >= w - k:
        x1 = p2[1] - p1[1] -(w-x)
    if y <= k:
        y1 = y
    elif y >= h - k:
        y1 = p2[0] - p1[0]-(h-y)
    he_pix = cdf[img[p1[0]+y1, p1[1] + x1]]
    return he_pix


def applyCLAHE(img, k):
    new_img = np.zeros_like(img, np.uint8)
    h, w = img.shape
    for i in range(0, h):
        for j in range(0, w):
            p1, p2 = get_tile(img, i, j, k)
            tile = img[p1[0]:p2[0], p1[1]:p2[1]]
            new_pix = applyHE(img, p1, p2, i, j, k, w, h)
            new_img[i, j] = new_pix
    return new_img


def clahe(img_path, window_size=150):
    """
    img_path    : Image to apply CLAHE algorithm\n
    window_size : Window size for the algorithm\n
    Returns target image as numpy array
    """
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    assert window_size <= min(h, w), "Window size is too large"
    k = window_size//2

    if img.ndim == 3:
        img_slices = [applyCLAHE(img[:, :, i], k) for i in range(3)]
        img_clahe = np.dstack(img_slices)
    elif img.ndim == 2:
        img_clahe = applyCLAHE(img, k)
    else:
        raise ValueError

    a,b = 1, 2
    if w >= 2*h:
        a, b = 2, 1

    plt.subplot(a, b, 1)
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Original Image')

    plt.subplot(a, b, 2)
    plt.imshow(img_clahe, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Contrast Limited Adaptive Histogram Equalized Image')
    plt.show()


if __name__ == "__main__":
    img_path = '../data/1.jpg'
    clahe(img_path)
