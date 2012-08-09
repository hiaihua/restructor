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
	cam.initGrabber(320, 240);

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

	}
}

//--------------------------------------------------------------
void testApp::draw(){
	undistorted.draw(0, 0);

	int currentX = 0;

	for (int i = 0; i < images.size(); i++) {
		ofImage& image = images[i];
		image.draw(currentX, undistorted.getHeight() + 5);
		currentX += image.getWidth() + 5;
	}
}

//--------------------------------------------------------------
void testApp::keyPressed  (int key){
	if (images.size() >= 2) {
		return;
	}
	ofImage image;
	image.clone(undistorted);
	images.push_back(image);
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

