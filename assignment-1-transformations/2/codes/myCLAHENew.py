import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_tile(img, y, x, k):
    h, w = img.shape
    p1 = (max(0, y-k), max(0, x-k))
    p2 = (min(y+k, h-1), min(x+k, w-1))
    return p1, p2


def get_cdf(img, x, k, w, prev_pdf=None, threshold=0.05):
    # print(x, k, w-k)
    if x == 0:
        pdf = np.histogram(img, 256, (0, 256))[0]
    elif x <= k:
        pdf1 = np.histogram(img[:, -1], 256, (0, 256))[0]
        pdf = prev_pdf + pdf1
    elif x >= w-k:
        pdf2 = np.histogram(img[:, 0], 256, (0, 256))[0]
        pdf = prev_pdf - pdf2
    else:
        pdf1 = np.histogram(img[:, -1], 256, (0, 256))[0]
        pdf2 = np.histogram(img[:, 0], 256, (0, 256))[0]
        pdf = prev_pdf + pdf1 - pdf2

    # Limiting Contrast
    prev_pdf = pdf
    pdf = pdf / (img.shape[0]*img.shape[1])
    pdf = np.clip(pdf, 0, threshold)
    extra_mass = 1 - np.sum(pdf)
    pdf += extra_mass/256
    cdf = 255 * pdf.cumsum()
    cdf = cdf.astype(np.uint8)
    return cdf, prev_pdf


def applyHE(img, y, x, k, w, h, prev_pdf, threshold):
    cdf, prev_pdf = get_cdf(img, x, k, w, prev_pdf, threshold)
    x1, y1 = k, k
    if x > k:
        img = img[:, 1:]
    if x <= k:
        x1 = x
    elif x >= w - k:
        x1 = img.shape[1]-(w-x)
    if y <= k:
        y1 = y
    elif y >= h - k:
        y1 = img.shape[0]-(h-y)
    he_pix = cdf[img[y1, x1]]
    return he_pix, prev_pdf


def applyCLAHE(img, k, threshold):
    new_img = np.zeros_like(img, np.uint8)
    prev_pdf = None
    h, w = img.shape
    for i in range(0, h):
        for j in range(0, w):
            p1, p2 = get_tile(img, i, j, k)
            tile = img[p1[0]:p2[0], max(0, p1[1]-1):p2[1]]
            new_pix, prev_pdf = applyHE(tile, i, j, k, w, h, prev_pdf, threshold)
            # new_img[p1[0]:p2[0], p1[1]:p2[1]] = new_tile
            new_img[i, j] = new_pix
        print('{:.02f}{}'.format(100*(i+1)/h, '%'), end='\r')
    print('\nFinished a channel', flush=True)
    return new_img


def clahe(img_path, window_size, threshold):
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
        img_slices = [applyCLAHE(img[:, :, i], k, threshold) for i in range(3)]
        img_clahe = np.dstack(img_slices)
    elif img.ndim == 2:
        img_clahe = applyCLAHE(img, k, threshold)
    else:
        raise ValueError

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_clahe, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Contrast Limited Adaptive Histogram Equalized Image')
    plt.show()
    return img_clahe

if __name__ == "__main__":
    img_path = '../data/1.jpg'
    img_clahe = clahe(img_path, 300, 0.03)
    plt.imsave('clahe_img.jpg', img_clahe)
