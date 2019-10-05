import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def split(pdf, img_median):
    pdf_h1 = pdf[pdf <= (img_median)]
    pdf_h2 = pdf[pdf > (img_median)]
    return pdf_h1, pdf_h2


def get_cdf(pdf_new, img_median):
    cdf = 255 * pdf_new.cumsum()
    cdf = cdf.astype(np.uint8)
    return cdf


def applyHE(img):
    h, w = img.shape
    pdf = np.histogram(img, 256, (0, 256))[0]
    img_median = np.argsort(pdf)[len(pdf) // 2]
    h1, h2 = split(pdf, img_median)
    # h1 = np.array([0.5 / img_median] * len(h1))
    # h2 = np.array([0.5 / (256 - img_median)] * len(h2))
    pdf_new = np.array(list(h1) + list(h2))
    cdf = get_cdf(pdf_new, img_median)
    he_img = np.zeros_like(img, np.uint8)
    he_img = cdf[img]
    return he_img


def histogramEqualize(img_path):
    img = Image.open(img_path).convert('L')
    img = np.array(img)
    # img = img / 255.0

    if img.ndim == 3:
        img_slices = [applyHE(img[:, :, i]) for i in range(3)]
        he_img = np.dstack(img_slices)
    elif img.ndim == 2:
        he_img = applyHE(img)
    else:
        raise ValueError

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(he_img, cmap='gray')
    plt.title('Bi-Histogram Equalized Image')

    plt.subplot(2, 2, 3)
    plt.hist(img.ravel(), 64, (0, 256))
    plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.hist(he_img.ravel(), 64, (0, 256))
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    histogramEqualize('../data/barbara.png')
