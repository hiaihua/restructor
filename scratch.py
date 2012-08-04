#!/usr/bin/python
import cv
import os

def loadImages(dir, lst):
	with open(os.path.join(dir, lst)) as f:
		filenames = [each.strip() for each in f.readlines()]
		f.close()

	images = [cv.LoadImage(os.path.join(dir, filename), cv.CV_LOAD_IMAGE_COLOR) for filename in filenames]
	return images

def main():
	cv.NamedWindow('window', cv.CV_WINDOW_AUTOSIZE)
	images = loadImages("img", "list_of_photos.ifl")

	for image in images:
		cv.ShowImage('window', image)
		cv.WaitKey(10000)

if __name__ == "__main__":
	main()

