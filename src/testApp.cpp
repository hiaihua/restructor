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
}

//--------------------------------------------------------------
void testApp::update(){
	cam.update();
	if (cam.isFrameNew()) {

		calibration.undistort(toCv(cam), toCv(undistorted));
		undistorted.update();

		const Mat& mat = toCv(undistorted);
		//Mat mat2(mat, )
		Mat gray;
		cvtColor(mat, gray, CV_BGR2GRAY);
		cv::goodFeaturesToTrack(gray, corners, 500, 0.01, 5);


	}
}

//--------------------------------------------------------------
void testApp::draw(){
	undistorted.draw(0, 0);

	ofPushStyle();
	ofNoFill();
	for (int i = 0; i < corners.size(); i++) {
		ofRect(corners[i].x, corners[i].y, 2, 2);
	}
	ofPopStyle();
}

//--------------------------------------------------------------
void testApp::keyPressed  (int key){ 
	
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

