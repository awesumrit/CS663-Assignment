import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def make_kernel(r):
    Y, X = np.mgrid[:2*r+1, :2*r+1]
    dist = (Y - r)**2 + (X - r)**2
    mask = dist <= r**2
    return mask


def multiply(x, y, h, w, r, img):
    kernel = make_kernel(r)
    img = img[max(0,y-r):min(y+r+1, h), max(0,x-r):min(w, x+r+1)]
    if x < r:
        kernel = kernel[:, r-x:]
    elif x >= w - r:
        kernel = kernel[:, :w-r-x-1]
    if y < r:
        kernel = kernel[r-y:, :]
    elif y >= h - r:
        kernel = kernel[:h-r-y-1, :]
    kernel =  kernel / np.sum(kernel)
    try:
        conv = kernel * img
    except:
        print(y, x, h, w, r)
        print(kernel.shape, img.shape)
    return np.round(np.sum(conv)).astype(np.int)


def equidistant_points(x, y, h, w, dist):
    points = set()
    for i in range(int(dist/1.414)+1):
        j = np.round((dist**2-i**2)**(0.5))
        points.add((j, i))
        points.add((i, j))
    points = np.array(list(points)).astype(np.int)
    p1 = points
    p2 = points * [-1, 1]
    p3 = points * [1, -1]
    p4 = points * [-1, -1]
    points = np.concatenate([p1, p2, p3, p4]) + [y, x]
    return points


def min_dist(x, y, mask, thresh):
    h, w = mask.shape
    low = 1
    high = thresh
    while True:
        if low >= thresh - 1:
            return thresh
        dist = int((low+high)/2)
        # print(low, dist, high)
        flag = False
        points = equidistant_points(x, y, h, w, dist)
        for p in points:
            if p[0] >= 0 and p[0] < h:
                if p[1] >= 0 and p[1] < w:
                    if mask[p[0], p[1]] > 0:
                        flag = True
                        break
        if flag:
            high = dist
        else:
            low = dist
        if high - low < 2:
            return dist

    
def blur(img, mask, r_map):
    h, w = mask.shape
    new_img = np.zeros((h,w), np.uint8)
    for y in range(h):
        for x in range(w):
            new_img[y,x] = multiply(x, y, h, w, r_map[y, x], img)
    new_img = img * (mask > 0) + new_img * (mask == 0)
    return new_img


def blur_helper(img, mask, r_map):
    blur_slices = [blur(img[:,:,i], mask, r_map) for i in range(3)]
    new_img = np.dstack(blur_slices)
    return new_img
