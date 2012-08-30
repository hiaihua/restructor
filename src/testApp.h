#ifndef _TEST_APP
#define _TEST_APP


#include "ofMain.h"
#include "ofxCv.h"
#ifdef TARGET_ANDROID
#include "ofxAndroidApp.h"
class testApp : public ofxAndroidApp {
#else
class testApp : public ofBaseApp {
#endif
	public:
		
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);

		virtual bool menuItemSelected(string menu_id_str);

		void doRestruct();
		void doCalibrate();
		void addPicture();

		ofVideoGrabber cam;
		ofImage undistorted;
		ofImage prevUndistorted;
		ofxCv::Calibration calibration;

		std::vector<ofImage> images;
		std::vector<ofImage> grayImages;
		ofPtr<ofxCv::Flow> flow;

		cv::Mat fundamentalMatrix;
		cv::Mat essentialMatrix;
		ofImage disparity;
	private:
		ofPtr<ofxCv::Flow> initializePyrLK();
		ofPtr<ofxCv::Flow> initializeFarneback();
		ofMesh mesh;
		std::vector<ofMatrix4x4> ofRs;
		ofEasyCam easycam;
		std::vector<cv::KeyPoint> correspImg1Pt;
		float computedAvgErr;
		bool calibrating;
		ofPixels previous;
		ofPixels diff;

		cv::Mat_<double> R;
		cv::SVD svd;

		float diffMean;
		float lastTime;

		void doCalibrationUpdate();

		std::vector<cv::KeyPoint> convertFrom(const std::vector<cv::Point2f>& points);
		double TriangulatePoints(const vector<cv::KeyPoint>& pt_set1,
							   const std::vector<cv::KeyPoint>& pt_set2,
							   const cv::Mat& Kinv,
							   const cv::Mat& distcoeff,
							   const cv::Matx34d& P,
							   const cv::Matx34d& P1,
							   std::vector<cv::Point3d>& pointcloud,
							   std::vector<cv::KeyPoint>& correspImg1Pt);
		cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
				cv::Matx34d P,			//camera 1 matrix
				cv::Point3d u1,			//homogenous image point in 2nd camera
				cv::Matx34d P1			//camera 2 matrix
													);
		cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
				cv::Matx34d P,		//camera 1 matrix
										   cv::Point3d u1,		//homogenous image point in 2nd camera
										   cv::Matx34d P1		//camera 2 matrix
										   );


};

#endif	

