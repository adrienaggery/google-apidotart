import cv2
import numpy as np
from numpy import ma


images = ['images/portrait_154984.jpg', 'images/portrait_272849.jpg', 'images/portrait_391404.jpg', 'images/portrait_512684.jpg']


BLEND_W = 400
N_LAYERS = len(images)

def makeGaussian(size, fwhm):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = size//2
    y0 = size//2
    g = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return g / np.max(g)

CIRCLE_WIDTH = 40
circle_mask = makeGaussian(CIRCLE_WIDTH, CIRCLE_WIDTH/2)
def dot(blend_mask, x, y, weight=2):
    blend_mask[y:y+CIRCLE_WIDTH, x:x+CIRCLE_WIDTH] -= circle_mask[:, :] * weight
    blend_mask = np.clip(blend_mask, 0, N_LAYERS - 1)
    return blend_mask

def create_blend_mask(weight=1):
    blend_mask = np.ones((BLEND_W, BLEND_W)) * (N_LAYERS - 1)

    P = 20
    def arc(blend_mask, x, y, weight=1):
        for i in xrange(1, P):
            blend_mask = dot(blend_mask, x + 5 * i + P/2 * np.sin(np.pi * i / float(P)), y + 5 * i + P/2 * np.cos(np.pi * i / float(P)), weight=weight)
        return blend_mask

    blend_mask = arc(blend_mask, 40, 40, weight=weight)
    blend_mask = arc(blend_mask, 70, 50, weight=weight)
    blend_mask = arc(blend_mask, 50, 100, weight=weight)
    blend_mask = arc(blend_mask, 40, 80, weight=weight)

    return blend_mask

def crop_image(im):
    # resize to BLEND_W if <= BLEND_W
    return im[:BLEND_W, :BLEND_W, :]

def create_blend(cropped_images, blend_mask):
    blend = np.zeros((BLEND_W, BLEND_W, 3)) # resulting blended image

    for i in xrange(N_LAYERS - 1):
        u = cropped_images[i]
        v = cropped_images[i + 1]
        m = np.ma.masked_outside(blend_mask, i, i+1) - i
        s = 1 - m
        blend[:, :, 0] += u[:, :, 0] * s + v[:, :, 0] * m
        blend[:, :, 1] += u[:, :, 1] * s + v[:, :, 0] * m
        blend[:, :, 2] += u[:, :, 2] * s + v[:, :, 0] * m
    return np.clip(blend, 0, 255).astype(np.uint8)

blend_mask = create_blend_mask(0.4)
#cv2.imshow('blend_mask', np.clip(blend_mask/np.max(blend_mask) * 256, 0, 255).astype(np.uint8))

im = [cv2.imread(im) for im in images]
cropped_im = [crop_image(im) for im in im]
blend = create_blend(cropped_im, blend_mask)
cv2.imshow('im', blend)

cv2.waitKey(0)
