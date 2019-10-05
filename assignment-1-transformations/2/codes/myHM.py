import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def get_cdf(img):
    img = img.reshape(-1)
    img = img[np.where(img>=0)]
    pdf = np.histogram(img, 256, (0, 256))[0] / (len(img))
    cdf = pdf.cumsum()
    cdf = 255 * (cdf - cdf.min())/(cdf.max() - cdf.min())
    cdf = cdf.astype(np.uint8)
    return cdf


def applyHE(img, mask=False):
    cdf = get_cdf(img)
    if mask:
        img = img * (img >= 0)
    he_img = np.zeros_like(img, np.uint8)
    he_img = cdf[img]
    return he_img


def applyHM(src, ref):
    ref_cdf = get_cdf(ref)
    src_he = applyHE(src, mask=True)
    inv_cdf = np.zeros(256, np.int) - 1
    for i, p in enumerate(ref_cdf):
        if p == -1:
            inv_cdf[p] = i
    for i, p in enumerate(inv_cdf):
        if p == -1:
            min_diff = 1000
            for j, p1 in enumerate(ref_cdf):
                if abs(i - p1) < min_diff:
                    min_diff = abs(i - p1)
                    inv_cdf[i] = j
    inv_cdf = inv_cdf.astype(np.uint8)
    hm_img = np.zeros_like(src, np.uint8)
    hm_img = inv_cdf[src_he]
    return hm_img, src_he


def histogramMatch(src_path, ref_path, src_mask, ref_mask):
    src = Image.open(src_path)
    src = np.array(src)
    ref = Image.open(ref_path)
    ref = np.array(ref)

    src_mask = np.array(Image.open(src_mask))
    src_mask = np.tile(src_mask, (3,1,1))
    src_mask = np.transpose(src_mask, (1,2,0))

    ref_mask = np.array(Image.open(ref_mask))
    ref_mask = np.tile(ref_mask, (3,1,1))
    ref_mask = np.transpose(ref_mask, (1,2,0))

    src_m = src * src_mask + (src_mask - 1)
    ref_m = ref * ref_mask + (ref_mask - 1)

    img_slices = [applyHM(src_m[:, :, i], ref_m[:, :, i])[0] for i in range(3)]
    hm_img = np.dstack(img_slices)
    hm_img_m = hm_img * src_mask + (src_mask - 1)

    he_slices = [applyHM(src_m[:, :, i], ref_m[:, :, i])[1] for i in range(3)]
    he_img = np.dstack(he_slices)
    print(he_img.shape)
    he_img_m = he_img * src_mask + (src_mask - 1)

    plt.subplot(2, 4, 1)
    plt.imshow(src, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Source Image')

    plt.subplot(2, 4, 2)
    plt.imshow(ref, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Reference Image')

    plt.subplot(2, 4, 3)
    plt.imshow(he_img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Histogram Equalized Image')

    plt.subplot(2, 4, 4)
    plt.imshow(hm_img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title('Histogram Matched Image')

    plt.subplot(2, 4, 5)
    plt.hist(src_m.ravel(), 64, (0, 256))
    plt.yticks([])

    plt.subplot(2, 4, 6)
    plt.hist(ref_m.ravel(), 64, (0, 256))
    plt.yticks([])

    plt.subplot(2, 4, 7)
    plt.hist(he_img_m.ravel(), 64, (0, 256))
    plt.yticks([])

    plt.subplot(2, 4, 8)
    plt.hist(hm_img_m.ravel(), 64, (0, 256))
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    histogramMatch('../data/retina.png', '../data/retinaRef.png',
                   '../data/retinaMask.png', '../data/retinaRefMask.png')
