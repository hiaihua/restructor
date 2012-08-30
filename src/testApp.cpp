#include "testApp.h"
#include "ofxCv/Calibration.h"
#include "ofxCv/Helpers.h"
#include "ofFileUtils.h"

using namespace ofxCv;
using namespace cv;
using namespace std;

const float diffThreshold = 2.5; // maximum amount of movement
const float timeThreshold = 1; // minimum time between snapshots
const int startCleaning = 10; // start cleaning outliers after this many samples

//--------------------------------------------------------------
void testApp::setup(){
	ofSetVerticalSync(true);
	cam.initGrabber(640, 480);

	ofSetLogLevel(OF_LOG_NOTICE);
	//ofSetOrientation(OF_ORIENTATION_90_LEFT);
	// TODO: go through calibration setups if xml does not exist
	calibration.setFillFrame(true);

	FileStorage settings(ofToDataPath("settings.yml"), FileStorage::READ);
	if(settings.isOpened()) {
		int xCount = settings["xCount"], yCount = settings["yCount"];
		calibration.setPatternSize(xCount, yCount);
		float squareSize = settings["squareSize"];
		calibration.setSquareSize(squareSize);
		CalibrationPattern patternType;
		switch(settings["patternType"]) {
			case 0: patternType = CHESSBOARD; break;
			case 1: patternType = CIRCLES_GRID; break;
			case 2: patternType = ASYMMETRIC_CIRCLES_GRID; break;
		}
		calibration.setPatternType(patternType);
	}

	calibration.load("calibration.yml");

	imitate(undistorted, cam);
	imitate(previous, cam);
	imitate(diff, cam);

	//flow = initializeFarneback();
	flow = initializePyrLK();


	mesh.setMode(OF_PRIMITIVE_POINTS);

	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(3);

	easycam.setDistance(50);

	computedAvgErr = 0;
	calibrating = false;

	diffMean = 0;
	lastTime = 0;
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

void testApp::doCalibrationUpdate() {
	Mat camMat = toCv(cam);
	Mat prevMat = toCv(previous);
	Mat diffMat = toCv(diff);

	absdiff(prevMat, camMat, diffMat);
	camMat.copyTo(prevMat);

	diffMean = mean(Mat(mean(diffMat)))[0];

	float curTime = ofGetElapsedTimef();
	if(curTime - lastTime > timeThreshold && diffMean < diffThreshold) {
		if(calibration.add(camMat)) {
			cout << "re-calibrating" << endl;
			calibration.calibrate();
			if(calibration.size() > startCleaning) {
				calibration.clean();
			}
			calibration.save("calibration.yml");
			lastTime = curTime;
		}
	}

	if(calibration.size() > 0) {
		calibration.undistort(toCv(cam), toCv(undistorted));
		undistorted.update();
	}

}

//--------------------------------------------------------------
void testApp::update(){
	cam.update();
	if (cam.isFrameNew()) {

		if (calibrating) {
			doCalibrationUpdate();
		}

		if (calibration.size() > 0) {
			calibration.undistort(toCv(cam), toCv(undistorted));
			undistorted.update();
		}
	}
}

//--------------------------------------------------------------
void testApp::draw(){
	ofPushStyle();
	std::stringstream ss;
	ss << "avg Err: " << computedAvgErr << std::endl;
	if (calibrating) {
		ss << "calibrating..." << std::endl;
	}
	ofSetHexColor(0xffffff);
	ofDrawBitmapString(ss.str(), 20, 20);
	ofNoFill();
	easycam.begin();
	ofRect(0, 0, 3, 3);
	ofScale(0.1, 0.1, 0.1);
	cam.draw(0, 0);

	mesh.draw();
	easycam.end();
	ofPopStyle();
}

std::vector<KeyPoint> testApp::convertFrom(const std::vector<Point2f>& points) {
	std::vector<KeyPoint> ret;

	for (int i = 0; i < points.size(); i++) {
		const Point2d& point = points[i];
		KeyPoint keyPoint(point.x, point.y, 1);
		ret.push_back(keyPoint);
	}

	return ret;
}

//--------------------------------------------------------------
void testApp::keyPressed  (int key){
	if (key == 'x') {
		doRestruct();
	}
	else if (key == 'c') {
		doCalibrate();
	}
	else if (key == 't') {
		addPicture();
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
	if (button == 4) {
		easycam.setDistance(easycam.getDistance() + 0.5);
	} else if (button == 3) {
		easycam.setDistance(easycam.getDistance() - 0.5);
	}
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> testApp::LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
								   Matx34d P,		//camera 1 matrix
								   Point3d u1,		//homogenous image point in 2nd camera
								   Matx34d P1		//camera 2 matrix
								   )
{

	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	//	cout << "u " << u <<", u1 " << u1 << endl;
	//	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u.x*P(1)-u.y*P(0);
	//	A(3) = u1.x*P1(2)-P1(0);
	//	A(4) = u1.y*P1(2)-P1(1);
	//	A(5) = u1.x*P(1)-u1.y*P1(0);
	//	Matx43d A; //not working for some reason...
	//	A(0) = u.x*P(2)-P(0);
	//	A(1) = u.y*P(2)-P(1);
	//	A(2) = u1.x*P1(2)-P1(0);
	//	A(3) = u1.y*P1(2)-P1(1);
	Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),
			  u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),
			  u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),
			  u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
			  );
	Matx41d B(-(u.x*P(2,3)	-P(0,3)),
			  -(u.y*P(2,3)	-P(1,3)),
			  -(u1.x*P1(2,3)	-P1(0,3)),
			  -(u1.y*P1(2,3)	-P1(1,3)));

	Mat_<double> X;
	solve(A,B,X,DECOMP_SVD);

	return X;
}

#define EPSILON 0.0001

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> testApp::IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
											Matx34d P,			//camera 1 matrix
											Point3d u1,			//homogenous image point in 2nd camera
											Matx34d P1			//camera 2 matrix
											) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4,1);
	for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
		Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,
				  (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,
				  (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,
				  (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
				  );
		Mat_<double> B = (Mat_<double>(4,1) <<	  -(u.x*P(2,3)	-P(0,3))/wi,
												  -(u.y*P(2,3)	-P(1,3))/wi,
												  -(u1.x*P1(2,3)	-P1(0,3))/wi1,
												  -(u1.y*P1(2,3)	-P1(1,3))/wi1
						  );

		solve(A,B,X_,DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
	}
	return X;
}

//Triagulate points
double testApp::TriangulatePoints(const vector<KeyPoint>& pt_set1,
					   const vector<KeyPoint>& pt_set2,
					   const Mat& Kinv,
					   const Mat& distcoeff,
					   const Matx34d& P,
					   const Matx34d& P1,
					   vector<Point3d>& pointcloud,
					   vector<KeyPoint>& correspImg1Pt)
{
#ifdef __SFM__DEBUG__
	vector<double> depths;
#endif

//	pointcloud.clear();
	correspImg1Pt.clear();

	Matx44d P1_(P1(0,0),P1(0,1),P1(0,2),P1(0,3),
				P1(1,0),P1(1,1),P1(1,2),P1(1,3),
				P1(2,0),P1(2,1),P1(2,2),P1(2,3),
				0,		0,		0,		1);
	Matx44d P1inv(P1_.inv());
	Mat_<double> K = Kinv.inv();

	cout << "Triangulating...";
	double t = getTickCount();
	vector<double> reproj_error;
	unsigned int pts_size = pt_set1.size();

#if 0
	//convert to Point2f
	vector<Point2f> _pt_set1_pt,_pt_set2_pt;
	KeyPointsToPoints(pt_set1,_pt_set1_pt);
	KeyPointsToPoints(pt_set2,_pt_set2_pt);

	//undistort
	Mat pt_set1_pt,pt_set2_pt;
	undistortPoints(_pt_set1_pt, pt_set1_pt, K, Mat());
	undistortPoints(_pt_set2_pt, pt_set2_pt, K, Mat());

	//triangulate
	Mat pt_set1_pt_2r = pt_set1_pt.reshape(1, 2);
	Mat pt_set2_pt_2r = pt_set2_pt.reshape(1, 2);
	Mat pt_3d_h(1,pts_size,CV_32FC4);
	cv::triangulatePoints(P,P1,pt_set1_pt_2r,pt_set2_pt_2r,pt_3d_h);

	//calculate reprojection
	vector<Point3f> pt_3d;
	convertPointsHomogeneous(pt_3d_h.reshape(4, 1), pt_3d);
	cv::Mat_<double> R = (cv::Mat_<double>(3,3) << P(0,0),P(0,1),P(0,2), P(1,0),P(1,1),P(1,2), P(2,0),P(2,1),P(2,2));
	Vec3d rvec; Rodrigues(R ,rvec);
	Vec3d tvec(P(0,3),P(1,3),P(2,3));
	vector<Point2f> reprojected_pt_set1;
	projectPoints(pt_3d,rvec,tvec,K,distcoeff,reprojected_pt_set1);

	for (unsigned int i=0; i<pts_size; i++) {
		CloudPoint cp;
		cp.pt = pt_3d[i];
		pointcloud.push_back(cp);
		reproj_error.push_back(norm(_pt_set1_pt[i]-reprojected_pt_set1[i]));
	}
#else
#pragma omp parallel for num_threads(1)
	for (int i=0; i<pts_size; i++) {
		Point2f kp = pt_set1[i].pt;
		Point3d u(kp.x,kp.y,1.0);
		Mat_<double> um = Kinv * Mat_<double>(u);
		u.x = um(0); u.y = um(1); u.z = um(2);

		Point2f kp1 = pt_set2[i].pt;
		Point3d u1(kp1.x,kp1.y,1.0);
		Mat_<double> um1 = Kinv * Mat_<double>(u1);
		u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);

		Mat_<double> X = IterativeLinearLSTriangulation(u,P,u1,P1);

//		cout << "3D Point: " << X << endl;
//		Mat_<double> x = Mat(P1) * X;
//		cout <<	"P1 * Point: " << x << endl;
//		Mat_<double> xPt = (Mat_<double>(3,1) << x(0),x(1),x(2));
//		cout <<	"Point: " << xPt << endl;
		Mat_<double> xPt_img = K * Mat(P1) * X;
//		cout <<	"Point * K: " << xPt_img << endl;
		Point2f xPt_img_(xPt_img(0)/xPt_img(2),xPt_img(1)/xPt_img(2));

#pragma omp critical
		{
			reproj_error.push_back(norm(xPt_img_-kp1));

			Point3d pt = Point3d(X(0),X(1),X(2));

			pointcloud.push_back(pt);
			correspImg1Pt.push_back(pt_set1[i]);
#ifdef __SFM__DEBUG__
			depths.push_back(X(2));
#endif
		}
	}
#endif

	Scalar mse = mean(reproj_error);
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Done. ("<<pointcloud.size()<<"points, " << t <<"s, mean reproj err = " << mse[0] << ")"<< endl;

	//show "range image"
#ifdef __SFM__DEBUG__
	{
		double minVal,maxVal;
		minMaxLoc(depths, &minVal, &maxVal);
		Mat tmp(240,320,CV_8UC3,Scalar(0,0,0)); //cvtColor(img_1_orig, tmp, CV_BGR2HSV);
		for (unsigned int i=0; i<pointcloud.size(); i++) {
			double _d = MAX(MIN((pointcloud[i].z-minVal)/(maxVal-minVal),1.0),0.0);
			circle(tmp, correspImg1Pt[i].pt, 1, Scalar(255 * (1.0-(_d)),255,255), CV_FILLED);
		}
		cvtColor(tmp, tmp, CV_HSV2BGR);
		imshow("Depth Map", tmp);
		waitKey(0);
		destroyWindow("Depth Map");
	}
#endif

	return mse[0];
}
bool testApp::menuItemSelected(string menu_id_str) {
	if (menu_id_str == "calibrateOption") {
		doCalibrate();
	}
	else if(menu_id_str == "restructOption") {
		doRestruct();
	}
	else if (menu_id_str == "clearOption") {
		images.clear();
		mesh.clear();
	}
	else if (menu_id_str == "takePictureOption") {
		addPicture();
	}
	return true;
}

void testApp::doCalibrate() {
	calibrating = !calibrating;
}

void testApp::doRestruct() {
	std::vector<std::vector<cv::Point2f> > points;
	points.push_back(flow->getPointsPrev());
	points.push_back(flow->getPointsNext());

	cv::Mat k = (cv::Mat)calibration.getDistortedIntrinsics().getCameraMatrix();
	cv::Mat distCoeffs = calibration.getDistCoeffs();

	std::cout << "R: " << (cv::Mat)R << std::endl;
	//cv::Mat_<double> t = svd.u.col(2); //u3
	Matx33d Z(0, -1, 0,
			1, 0, 0,
			0, 0, 0);
	cv::Mat_<double> t = svd.vt.t() * Mat(Z) * svd.vt;
	std::cout << "t: " << (cv::Mat)t << std::endl;

	Vec3d tVec = Vec3d(t(1,2), t(2,0), t(0,1));

	Matx34d P = Matx34d(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0);
	Matx34d P1 = Matx34d(R(0,0),    R(0,1), R(0,2), tVec(0),
			R(1,0),    R(1,1), R(1,2), tVec(1),
			R(2,0),    R(2,1), R(2,2), tVec(2));

	vector<Point3d> pointcloud;
	TriangulatePoints(convertFrom(points[0]), convertFrom(points[1]), k.inv(),
			distCoeffs, P, P1, pointcloud, correspImg1Pt);

	mesh.clear();

	for (int i = 0; i < pointcloud.size(); i++) {
		const Point3d& point = pointcloud[i];
		ofVec3f vec(point.x, point.y, point.z);
		mesh.addColor(ofColor::blue);
		mesh.addVertex(vec);
	}
}

void testApp::addPicture() {
	if (images.size() >= 8) {
		return;
	}

	if (undistorted.height < 480 || undistorted.width < 640) {
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

	if (images.size() >= 2) {
		const ofImage& image1 = images[images.size() - 2];
		const ofImage& image2 = images[images.size() - 1];

		std::vector<std::vector<cv::Point2f> > points;
		points.push_back(flow->getPointsPrev());
		points.push_back(flow->getPointsNext());


		fundamentalMatrix = (cv::Mat)cv::findFundamentalMat(points[0], points[1]);
		cv::Mat cameraMatrix = (cv::Mat)calibration.getDistortedIntrinsics().getCameraMatrix();
		cv::Mat cameraMatrixInv = cameraMatrix.inv();
		std::cout << "fund: " << fundamentalMatrix << std::endl;

		std::vector<std::vector<Point3f> > lines(2);
		cv::computeCorrespondEpilines(points[0], 1, fundamentalMatrix, lines[0]);
		cv::computeCorrespondEpilines(points[1], 2, fundamentalMatrix, lines[1]);
		double avgErr = 0;

		for (int i = 0; i < points[0].size(); i++) {
			double err = fabs(points[0][i].x*lines[1][i].x +
					points[0][i].y*lines[1][i].y + lines[1][i].z)
					+ fabs(points[1][i].x*lines[0][i].x +
							points[1][i].y*lines[0][i].y + lines[0][i].z);
						        avgErr += err;
		}
		computedAvgErr = avgErr/(points[0].size());

		svd = cv::SVD(essentialMatrix);
		Matx33d W(0, -1, 0,
				1, 0, 0,
				0, 0, 1);
		R = svd.u * Mat(W).inv() * svd.vt; //HZ 9.19






		for (int y = 0; y < image1.height; y += 10) {
			for (int x = 0; x < image1.width; x += 10) {
				Vec3d vec(x, y, 0);

				Point3d point1(vec.val[0], vec.val[1], vec.val[2]);
				Vec3d result = (cv::Mat)((cv::Mat)R * (cv::Mat)vec);
				Point3d point2 = result;


				mesh.addColor(image1.getColor(x, y));
				mesh.addVertex(ofVec3f(point1.x, point1.y, point1.z));

				mesh.addColor(image2.getColor(x, y));
				mesh.addVertex(ofVec3f(point2.x, point2.y, point2.z));
			}
		}
	}

}
