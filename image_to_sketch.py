import numpy as np
import imageio
import matplotlib.pyplot as plt
import scipy.ndimage

img = imageio.imread('images/image_to_sketch.jpg')
img = img.shape(196, 160, 30)


# Gray scale
def gray_scale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


gray_img = gray_scale(img)

# Invert image

inverted_img = 255 - gray_img

blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=5)


def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')


final_img = dodge(blur_img, gray_img)

plt.imshow(final_img, cmap='gray')
plt.imsave('img2.png', final_img, cmap='gray', vmin=0, vmax=255)
