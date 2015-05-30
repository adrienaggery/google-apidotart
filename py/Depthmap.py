import os, sys, cv2, numpy
import time
import redis

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/lib')
import Leap

img_layer = 4
last_time = 0


MAP_WIDTH = 180
CIRCLE_SIZE = 20

r = redis.Redis()

def makeGaussian(size, fwhm):
	x = numpy.arange(0, size, 1, float)
	y = x[:,numpy.newaxis]
	x0 = size // 2
	y0 = size // 2
	return numpy.exp(-4 * numpy.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

mask_map = numpy.ones((MAP_WIDTH, MAP_WIDTH))
hand_matrix = makeGaussian(CIRCLE_SIZE, CIRCLE_SIZE/2) * 0.005


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
            px = cx - CIRCLE_SIZE/2
            py = cy - CIRCLE_SIZE/2
            if (px >= 0 and px <= MAP_WIDTH - CIRCLE_SIZE and py >= 0 and py < MAP_WIDTH - CIRCLE_SIZE):
                mask_map[py:py + CIRCLE_SIZE, px:px + CIRCLE_SIZE] -= hand_matrix * strength
                mask_map = numpy.clip(mask_map, 0, img_layer - 1)
                if time.time() - last_time >= 0.1:
                    last_time = time.time()
                    r['mask'] = mask_map.tostring()
                    r['lastx'] = cx
                    r['lasty'] = cy

class Depthmap():
	def __init__(self):
		# Leap init
		self.listener = LeapListener()
		self.controller = Leap.Controller()
		self.controller.add_listener(self.listener)
