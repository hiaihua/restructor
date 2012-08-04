#!/usr/bin/python
import cv2
import cv
import os
import yaml
def loadImages(dir, lst):
	with open(os.path.join(dir, lst)) as f:
		filenames = [each.strip() for each in f.readlines()]
		f.close()

	images = [cv.LoadImage(os.path.join(dir, filename), cv.CV_LOAD_IMAGE_COLOR) for filename in filenames]
	return images

class OpenCVLoader(yaml.SafeLoader):
	def construct_opencv_matrix(loader, suffix, node):
		return loader.construct_yaml_map(node)

OpenCVLoader.add_multi_constructor(u"tag:yaml.org,2002:opencv-matrix", OpenCVLoader.construct_opencv_matrix)

class Intrinsics:
	#TODO
	def __init__(self, cameraMatrix, imageSize, sensorSize):
		self.cameraMatrix = cameraMatrix
		self.imageSize = imageSize
		self.sensorSize = sensorSize

		fovx, fovy, focalLength, principalPoint, aspectRatioCalibration = cv2.CalibrationMatrixValues(cameraMatrix, imageSize, sensorSize.width, sensorSize.height)
		self.fov = (fovx, fovy)
		self.focalLength = focalLength
		self.principalPoint = principalPoint
		self.aspectRatioCalibration = aspectRatioCalibration

class Calibration:
	def __init__(self, dir, filename):
		self.cameraMatrix = cv.Load(os.path.join(dir, filename))
		with open(os.path.join(dir, filename)) as f:
			lines = f.readlines()
			lines = lines[1:]
			yamlData = "".join(lines)
			obj = yaml.load(yamlData, Loader=OpenCVLoader)
			
			self.imageSize = (obj["imageSize_width"], obj["imageSize_height"])
			self.sensorSize = (obj["sensorSize_width"], obj["sensorSize_height"])
			self.distCoeffs = obj["distCoeffs"]
			self.reprojectionError = obj["reprojectionError"]
			self.features = obj["features"]
		self.distortedIntrinsics = Intrinsics(self.cameraMatrix, self.imageSize, self.sensorSize)
		self.updateUndistortion()

	def updateUndistortion(self):
		#TODO

def main():
	cv.NamedWindow('window', cv.CV_WINDOW_AUTOSIZE)
	images = loadImages("img", "list_of_photos.ifl")
	calibration = Calibration("", "calibration.yml")

	for image in images:
		cv2.ShowImage('window', image)
		cv2.WaitKey(10000)

if __name__ == "__main__":
	main()

