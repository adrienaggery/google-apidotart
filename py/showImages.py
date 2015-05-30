import cv2
import numpy as np
from numpy import ma
import time
import redis

r = redis.Redis()

images = ['images/portrait_154984.jpg', 'images/portrait_272849.jpg', 'images/portrait_391404.jpg', 'images/portrait_512684.jpg']
images = ['images/Homme{}.jpg'.format(i) for i in xrange(1, 5)]
images.reverse()

BLEND_W = 180

HEIGHT = None
WIDTH = None
N_LAYERS = len(images)

# Get min height and width, then crop
for filename in images:
    im = cv2.imread(filename)
    if HEIGHT is None:
        HEIGHT = im.shape[0]
        WIDTH = im.shape[1]
    else:
        HEIGHT = min(HEIGHT, im.shape[0])
        WIDTH = min(WIDTH, im.shape[1])

SCALE = 2
H_HEIGHT = HEIGHT / SCALE
H_WIDTH = WIDTH / SCALE


def crop_image(im):
    im = im[:SCALE * H_HEIGHT, :SCALE * H_WIDTH, :]
    return cv2.resize(im, (H_WIDTH, H_HEIGHT))


def create_blend(cropped_images, blend_mask):
    blend = np.zeros((H_HEIGHT, H_WIDTH, 3)) # resulting blended image

    for i in xrange(N_LAYERS - 1):
        u = cropped_images[i]
        v = cropped_images[i + 1]
        m = np.ma.masked_outside(blend_mask, i, i+1) - i
        s = 1 - m
        blend[:, :, 0] += u[:, :, 0] * s + v[:, :, 0] * m
        blend[:, :, 1] += u[:, :, 1] * s + v[:, :, 1] * m
        blend[:, :, 2] += u[:, :, 2] * s + v[:, :, 2] * m
    return np.clip(blend, 0, 255).astype(np.uint8)


cropped_im = [crop_image(cv2.imread(filename)) for filename in images]
prev_time = 0
while True:
    while time.time() - prev_time < 0.1:
        continue
    prev_time = time.time()
    blend_mask = np.fromstring(r['mask']).reshape((BLEND_W, BLEND_W)) * (N_LAYERS - 1)
    blend_mask = cv2.resize(blend_mask, (H_WIDTH, H_HEIGHT))
    blend = create_blend(cropped_im, blend_mask)
    x = int(float(r['lastx']) * H_WIDTH / BLEND_W)
    y = int(float(r['lasty']) * H_HEIGHT / BLEND_W)
    cv2.circle(blend, (x-5, y-5), 5, (255, 0, 0))
    cv2.imshow('im', cv2.resize(blend, (SCALE * H_WIDTH, SCALE * H_HEIGHT)))
    if not(cv2.waitKey(10)):
        break
