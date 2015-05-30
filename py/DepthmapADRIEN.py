import os, sys, cv2, numpy
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/lib')
import Leap

hand_range = 40
blur_range = 20
hand_speed_threshold = 0
map_size = [547, 650, 20]
img_layer = 8

def makeGaussian(size, fwhm):
	x = numpy.arange(0, size, 1, float)
	y = x[:,numpy.newaxis]
	x0 = size // 2
	y0 = size // 2
	return numpy.exp(-4 * numpy.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

mask_map = numpy.ones((map_size[0], map_size[1])) * (img_layer - 1)
hand_matrix = makeGaussian(hand_range, blur_range)

class LeapListener(Leap.Listener):
	def on_connect(self, controller):
		print "Leap connected"

	def on_frame(self, controller):
		global mask_map
		frame = controller.frame()
		if (len(frame.hands) == 1):
			if (numpy.absolute(frame.hands[0].palm_velocity.x) > hand_speed_threshold or numpy.absolute(frame.hands[0].palm_velocity.z > hand_speed_threshold)):
				pos = self.normalized_palm_position(frame)
				if (pos.x < map_size[0] - hand_range and pos.z < map_size[1] - hand_range):
					print "Depth updated"
					mask_map[pos.z:pos.z+hand_range, pos.x:pos.x+hand_range] -= hand_matrix
					mask_map = numpy.clip(mask_map, 0, img_layer - 1)
					m = (mask_map * 255.0 / (img_layer - 1)).astype(numpy.uint8)
					cv2.imshow("debug", m)

	def normalized_palm_position(self, frame):
		position = frame.interaction_box.normalize_point(frame.hands[0].stabilized_palm_position, True)
		position.x *= map_size[0]
		position.z *= map_size[1]
		position.y *= map_size[2]
		return position


class Depthmap():
	def __init__(self):
		# Leap init
		self.listener = LeapListener()
		self.controller = Leap.Controller()
		self.controller.add_listener(self.listener)
		cv2.namedWindow("debug")
