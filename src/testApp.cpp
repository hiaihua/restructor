#include "testApp.h"
#include "ofxCv/Calibration.h"
#include "ofxCv/Helpers.h"
#include "ofFileUtils.h"

using namespace ofxCv;
using namespace cv;
using namespace std;

//--------------------------------------------------------------
void testApp::setup(){
	ofSetVerticalSync(true);
	cam.initGrabber(640, 480);

	ofSetLogLevel(OF_LOG_NOTICE);
	//ofSetOrientation(OF_ORIENTATION_90_LEFT);

	// TODO: go through calibration setups if xml does not exist
	calibration.setFillFrame(true);
	calibration.load("calibration.yml");
	imitate(undistorted, cam);

	flow = initializeFarneback();

}

ofPtr<ofxCv::Flow> testApp::initializePyrLK() {
	ofxCv::FlowPyrLK* flow = new ofxCv::FlowPyrLK();

	flow->setMaxFeatures(200);
	flow->setQualityLevel(0.01);
	flow->setMinDistance(4);
	flow->setWindowSize(32);
	flow->setMaxLevel(3);

	return ofPtr<ofxCv::Flow>(flow);
}

ofPtr<ofxCv::Flow> testApp::initializeFarneback() {
	ofxCv::FlowFarneback* flow = new ofxCv::FlowFarneback();

	flow->setPyramidScale(0.5);
	flow->setNumLevels(4);
	flow->setWindowSize(8);
	flow->setNumIterations(2);

	flow->setPolyN(7);
	flow->setPolySigma(1.5);
	flow->setUseGaussian(false);

	return ofPtr<ofxCv::Flow>(flow);
}

//--------------------------------------------------------------
void testApp::update(){
	cam.update();
	if (cam.isFrameNew()) {

		calibration.undistort(toCv(cam), toCv(undistorted));
		undistorted.update();
	}
}

//--------------------------------------------------------------
void testApp::draw(){
	ofScale(0.5, 0.5, 1);
	cam.draw(0, 0);
	undistorted.draw(cam.getWidth() + 5, 0);

	int currentX = 0;

	for (int i = 0; i < images.size(); i++) {
		ofImage& image = images[i];
		image.draw(currentX, undistorted.getHeight() + 5);

		if (i == 1) {
			//flow->draw(currentX, undistorted.getHeight() + 5);
		}

		currentX += image.getWidth() + 5;
	}
	disparity.draw((undistorted.getHeight() + 5) * 2, 0);
}

//--------------------------------------------------------------
void testApp::keyPressed  (int key){
	if (images.size() >= 2) {
		return;
	}
	ofImage image;
	image.clone(undistorted);
	images.push_back(image);

	ofImage gray;
	gray.clone(undistorted);
	gray.setImageType(OF_IMAGE_GRAYSCALE);
	grayImages.push_back(gray);


	flow->calcOpticalFlow(image);

	if (images.size() == 2) {
		std::vector<cv::Point2f> points1 = flow->getPointsPrev();
		std::vector<cv::Point2f> points2 = flow->getPointsNext();


		cv::findFundamentalMat(points1, points2, fundamentalMatrix);

		disparity.clone(grayImages[0]);
		//cv::computeCorrespondEpilines(points1, 1, fundamentalMatrix, epilines);
		stereoBM(toCv(grayImages[0]), toCv(grayImages[1]), toCv(disparity));
	}
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){ 
	
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
	
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

