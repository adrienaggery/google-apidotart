# encoding: utf-8
import os, sys, cv2
import numpy as np
import time
import audio
import random
import requests
import json

p = audio.Player()
p.start()
p.play(20)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/lib')
import Leap

images = ['images/morpion{}.jpg'.format(i) for i in xrange(1, 4)]
images.reverse()

fready = 1
APIKEY = '867c6f9152a839360b3619a4b11eb87de567e86b9184627016ced9a4bdb7091e'
keywords = ['écorché', 'crâne humain (représenté)', 'allégorie de la Mort', 'autoportrait', 'voile (tissu)', 'fantôme', 'Narcisse (mythologie)', 'miroir', 'arbre mort']

def generateImages(apikey):
    images = {}
    for keyword in keywords:
        req = requests.get('http://api.art.rmngp.fr:80/v1/works', params={'facets[keywords]': keyword, 'lang': 'fr'}, headers={'ApiKey': apikey})
        if (req.status_code == 200):
            rep = req.json()
            json.dump(rep, open('rep.json', 'w'))
            hits = rep['hits']['hits']
            works = [{'id': hit['_id'], 'url': hit['_source']['images'][0]['urls']['large']['url']} for hit in hits]
            images[keyword] = works
            for work in works:
                import os
                if not(os.path.exists('images/im-{}.jpg'.format(work['id']))):
                    print 'download {}'.format(work['id'])
                    r = requests.get(work['url'])
                    if r.status_code == 200:
                        path = 'images/im-{}.jpg'.format(work['id'])
                        with open(path, 'wb') as f:
                            for chunk in r.iter_content(1024):
                                f.write(chunk)
    return images
imageDict = generateImages(APIKEY)

T_W = 210
T_H = 165

im = np.zeros((3 * T_H, 3 * T_W, 3), dtype=np.uint8)
for i, keyword in enumerate(keywords):
    x = i % 3
    y = i // 3
    work = random.choice(imageDict[keyword])
    work_im = cv2.imread('images/im-{}.jpg'.format(work['id']))
    (h, w, c) = work_im.shape
    if T_H / float(h) < T_W / float(w):
        r = T_W / float(w)
    else:
        r = T_H / float(h)
    work_im = cv2.resize(work_im, (int(r * w), int(r * h)))
    work_im = work_im[:T_H, :T_W, :]
    (h, w, c) = work_im.shape
    im[y * T_H:(y + 1) * h, x * T_W:(x + 1) * w] = work_im
imBlended = im


HEIGHT = None
WIDTH = None
N_LAYERS = len(images) + 1

MAP_WIDTH = 180
CIRCLE_SIZE = 40

# Get min height and width, then crop
for filename in images:
    im = cv2.imread(filename)
    if HEIGHT is None:
        HEIGHT = im.shape[0]
        WIDTH = im.shape[1]
    else:
        HEIGHT = min(HEIGHT, im.shape[0])
        WIDTH = min(WIDTH, im.shape[1])

UP_SCALE = 1
SCALE = 2
H_HEIGHT = HEIGHT / SCALE
H_WIDTH = WIDTH / SCALE

def crop_image(im):
    im = im[:SCALE * H_HEIGHT, :SCALE * H_WIDTH, :]
    return cv2.resize(im, (H_WIDTH, H_HEIGHT))

images = [cv2.imread(filename) for filename in images]
images = [imBlended] + images
cropped_im = [crop_image(im) for im in images]

def create_blend(cropped_images, blend_mask):
    blend = np.zeros((H_HEIGHT, H_WIDTH, 3)) # resulting blended image

    for i in xrange(N_LAYERS - 1):
        u = cropped_images[i]
        v = cropped_images[i + 1]
        m = np.ma.masked_outside(blend_mask, i, i + 1) - i
        s = 1 - m
        blend[:, :, 0] += u[:, :, 0] * s + v[:, :, 0] * m
        blend[:, :, 1] += u[:, :, 1] * s + v[:, :, 1] * m
        blend[:, :, 2] += u[:, :, 2] * s + v[:, :, 2] * m
    return np.clip(blend, 0, 255).astype(np.uint8)

def makeGaussian(size, fwhm):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = size // 2
    y0 = size // 2
    m = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
    return m / np.max(m)


mask_map = np.ones((MAP_WIDTH, MAP_WIDTH))
hand_matrix = makeGaussian(CIRCLE_SIZE, CIRCLE_SIZE / 2) / N_LAYERS / 40
last_time = 0

last_audio_time = 0
play_chord = 0.0
next_delta = 1.0
depth = 0


def play_sound(darkness):
    global last_audio_time
    global play_chord
    global next_delta
    depth = 1 + int(min(1, darkness + 0.5 * random.random()) * 10)
    if time.time() - last_audio_time >= next_delta:
        last_audio_time = time.time()
        next_delta = 0.8 + random.random() * 0.2
        n = 26 + random.randint(0, 4) - depth
        m = None
        play_chord += (random.random() - 0.5) * 0.25
        play_chord = max(0, min(1, play_chord))
        if play_chord < 0.5:
            m = 36 + random.randint(0, 4) - depth
        p.play(n, m)


class LeapListener(Leap.Listener):
    def on_connect(self, controller):
        print "Leap connected"

    def on_frame(self, controller):
        global mask_map
        global last_time
        frame = controller.frame()
        if (len(frame.hands) == 1):
            strength = 1 + 2 * frame.hands[0].grab_strength
            pos = frame.interaction_box.normalize_point(frame.hands[0].stabilized_palm_position, True)
            cx = int(pos.x * MAP_WIDTH)
            cy = int(pos.z * MAP_WIDTH)
            px = cx - CIRCLE_SIZE / 2
            py = cy - CIRCLE_SIZE / 2

            if (px >= 0 and px <= MAP_WIDTH - CIRCLE_SIZE and py >= 0 and py < MAP_WIDTH - CIRCLE_SIZE):
                mask_map[py:py + CIRCLE_SIZE, px:px + CIRCLE_SIZE] -= hand_matrix * strength
                darkness = max(0, min(1, 1.0 - np.exp(10 * np.log(mask_map.mean()))))
                play_sound(darkness)
                if time.time() - last_time >= 0.01:
                    last_time = time.time()
                    blend_mask = np.clip(mask_map, 0, 1.0) * (N_LAYERS - 1)
                    blend_mask = cv2.resize(blend_mask, (H_WIDTH, H_HEIGHT))
                    blend = create_blend(cropped_im, blend_mask)
                    x = int(cx * H_WIDTH / float(MAP_WIDTH))
                    y = int(cy * H_HEIGHT / float(MAP_WIDTH))
                    im = cv2.resize(blend, (UP_SCALE * SCALE * H_WIDTH, UP_SCALE * SCALE * H_HEIGHT))
                    cv2.circle(im, (UP_SCALE * SCALE * x, UP_SCALE * SCALE * y), UP_SCALE * SCALE * 5, (255, 0, 0), 1)
                    cv2.imshow("Vanitas", im)
        if not(cv2.waitKey(10)):
            p.stop()
            import sys
            sys.exit(0)


class Depthmap():
    def __init__(self):
        # Leap init
        self.listener = LeapListener()
        self.controller = Leap.Controller()
        self.controller.add_listener(self.listener)
        cv2.namedWindow("Vanitas")
