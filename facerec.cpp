#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
using namespace std;

const size_t in_width = 300;
const size_t in_height = 300;
const Scalar meanVal(104.0, 177.0, 123.0);

dnn::Net nnet = dnn::readNetFromTorch("openface.nn4.small2.v1.t7");

Mat recognize(Mat face) {
	Mat inputBlob = dnn::blobFromImage(face, 1./255, Size(96,96), Scalar(), true, false);
	nnet.setInput(inputBlob);
	Mat feature = nnet.forward().clone();
	return feature;
}

int main(){
	String modelConfiguration = "deploy.prototxt.txt";
	String modelBinary = "res10_300x300_ssd_iter_140000.caffemodel";
	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);

	if (net.empty())
		exit(-1);
	VideoCapture cap("1.mp4");
	for (;;){
		Mat frame;
		cap >> frame;
		
		if (frame.empty()){
			waitKey();
			break;
		}

		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);

		Mat inputBlob = blobFromImage(frame, 1.0, Size(in_width, in_height), meanVal, false, false); //将 Mat 转为一批图像
		net.setInput(inputBlob, "data"); //设置网络的输入
		Mat detection = net.forward("detection_out");

		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		ostringstream ss;
		ss << "FPS: " << 1000 / time;
		putText(frame, ss.str(), Point(20, 20), 0.2, 0.5, Scalar(0, 0, 255));
		
		float threshold = 0.6;
		for (int i = 0; i < detectionMat.rows; i++){
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > threshold){
				int xmin = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int ymin = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xmax = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int ymax = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect face((int)xmin, (int)ymin,
					(int)(xmax - xmin),
					(int)(ymax - ymin));

				rectangle(frame, face, Scalar(0, 0, 255), 2);
			}
		}
		imshow("detections", frame);
		if (waitKey(1) >= 0) break;
	}
	return 0;
}