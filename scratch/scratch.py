#!/usr/bin/python
import cv2
import cv
import os
import yaml
import array
import inspect
def loadImages(dir, lst):
	with open(os.path.join(dir, lst)) as f:
		filenames = [each.strip() for each in f.readlines()]
		f.close()

	images = [cv.LoadImage(os.path.join(dir, filename), cv.CV_LOAD_IMAGE_COLOR) for filename in filenames]
	return images

class OpenCVLoader(yaml.SafeLoader):
	def construct_opencv_matrix(self, loader, node):
		value = self.construct_mapping(node)
		for each in node.value:
			if each[0].value == "data":
				value["data"] = each[1]

				break
		ret = cv.CreateMatHeader(value["cols"], value["rows"], cv.CV_64FC1)
		dataList = self.construct_sequence(value["data"])
		dataArray = array.array('d', dataList)
		cv.SetData(ret, dataArray)
		#TODO: use 'dt' for the data type
		return ret
	
		

OpenCVLoader.add_multi_constructor(u"tag:yaml.org,2002:opencv-matrix", OpenCVLoader.construct_opencv_matrix)

class Intrinsics:
	def setup(self, cameraMatrix, imageSize, sensorSize):
		self.cameraMatrix = cameraMatrix
		self.imageSize = imageSize
		self.sensorSize = sensorSize

		fovx, fovy, focalLength, principalPoint, aspectRatioCalibration = cv.CalibrationMatrixValues(cameraMatrix, imageSize, sensorSize[0], sensorSize[1])
		self.fov = (fovx, fovy)
		self.focalLength = focalLength
		self.principalPoint = principalPoint
		self.aspectRatioCalibration = aspectRatioCalibration

class Calibration:
	def __init__(self, dir, filename):
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
			self.cameraMatrix = obj["cameraMatrix"]

		self.undistortMapX = cv.CreateMatHeader(self.imageSize[0], self.imageSize[1], cv.CV_32FC1)
		self.undistortMapY = cv.CreateMatHeader(self.imageSize[0], self.imageSize[1], cv.CV_32FC1)

		self.distortedIntrinsics = Intrinsics()
		self.undistortedIntrinsics = Intrinsics()
		self.distortedIntrinsics.setup(self.cameraMatrix, self.imageSize, self.sensorSize)
		self.fillFrame = True
		self.updateUndistortion()

	def updateUndistortion(self):
		if self.fillFrame:
			fillFrameAlpha = 0
		else:
			fillFrameAlpha = 1
		undistortedCameraMatrix = cv.CreateMat(self.distortedIntrinsics.cameraMatrix.width, self.distortedIntrinsics.cameraMatrix.height, cv.CV_64FC1)
		cv.GetOptimalNewCameraMatrix(self.distortedIntrinsics.cameraMatrix, self.distCoeffs, self.distortedIntrinsics.imageSize, fillFrameAlpha, undistortedCameraMatrix)
		cv2.initUndistortRectifyMap(self.distortedIntrinsics.cameraMatrix, self.distCoeffs, cv.CreateMatHeader(3, 3, cv.CV_64FC1), undistortedCameraMatrix, self.distortedIntrinsics.imageSize, cv.CV_16SC2, self.undistortMapX, self.undistortMapY)
		self.undistortedIntrinsics.setup(undistortedCameraMatrix, self.distortedIntrinsics.imageSize)

def main():
	cv.NamedWindow('window', cv.CV_WINDOW_AUTOSIZE)
	images = loadImages("img", "list_of_photos.ifl")
	calibration = Calibration("", "calibration.yml")

	for image in images:
		cv2.ShowImage('window', image)
		cv2.WaitKey(10000)

if __name__ == "__main__":
	main()

