import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from mySpatiallyVaryingKernel import min_dist, blur_helper, make_kernel


def make_rmap(img, mask, threshold):
    h, w = mask.shape
    r_map = np.zeros((h, w), np.int)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                r_map[y, x] = min_dist(x, y, mask, threshold)
        print('{:.02f}%'.format((y+1)*100/h), end='\r')
    print('\nRadius map calculated succesfully :)', flush='True')
    return r_map

print('\nRunning time of complete script ~ 6 mins\nBird.jpg consumes 90%% of this time\n')


mask1 = np.array(Image.open('../data/mask1.png').resize((212, 141)))
img1 = np.array(Image.open('../data/flower.jpg').resize((212, 141)))

mask2 = np.array(Image.open('../data/mask2.png').resize((550, 366)))
img2 = np.array(Image.open('../data/bird.jpg').resize((550, 366)))


# Part 1
plt.figure()
plt.subplot(231)
plt.imshow(mask1, cmap='gray')
plt.subplot(232)
img_masked1 = np.dstack([(mask1 > 0)*img1[:, :, i] for i in range(3)])
plt.imshow(img_masked1)
plt.subplot(233)
img_masked1 = np.dstack([(mask1 == 0)*img1[:, :, i] for i in range(3)])
plt.imshow(img_masked1)
plt.subplot(234)
plt.imshow(mask2, cmap='gray')
plt.subplot(235)
img_masked2 = np.dstack([(mask2 > 0)*img2[:, :, i] for i in range(3)])
plt.imshow(img_masked2)
plt.subplot(236)
img_masked2 = np.dstack([(mask2 == 0)*img2[:, :, i] for i in range(3)])
plt.imshow(img_masked2)
plt.show()

# Part 2
r_map1 = make_rmap(img1, mask1, 10)
r_map2 = make_rmap(img2, mask2, 20)

plt.figure()
plt.subplot(231)
plt.imshow(mask1, cmap='gray')
plt.subplot(232)
plt.imshow(img1)
plt.subplot(233)
plt.imshow(r_map1, cmap='jet')
plt.subplot(234)
plt.imshow(mask2, cmap='gray')
plt.subplot(235)
plt.imshow(img2)
plt.subplot(236)
plt.imshow(r_map2, cmap='jet')
plt.show()



# Part 3
# Using 20 pixels as threshold
plt.subplot(231)
kernel = make_kernel(4)
plt.imshow(kernel, cmap='gray')
plt.subplot(232)
kernel = make_kernel(8)
plt.imshow(kernel, cmap='gray')
plt.subplot(233)
kernel = make_kernel(12)
plt.imshow(kernel, cmap='gray')
plt.subplot(234)
kernel = make_kernel(16)
plt.imshow(kernel, cmap='gray')
plt.subplot(235)
kernel = make_kernel(20)
plt.imshow(kernel, cmap='gray')
plt.show()

# Part 4
print('Blurring image 1')
blur1 = blur_helper(img1, mask1, r_map1)
print('Blurring image 2')
blur2 = blur_helper(img2, mask2, r_map2)

plt.subplot(121)
plt.imshow(blur1)
plt.subplot(122)
plt.imshow(blur2)
plt.show()
