#!/usr/bin/python

import os, sys, Depthmap

def main():

	depthmap = Depthmap.Depthmap()

	try:
		sys.stdin.readline()
	except KeyboardInterrupt:
		pass

main()
