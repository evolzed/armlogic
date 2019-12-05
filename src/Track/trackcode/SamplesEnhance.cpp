#include "stdafx.h"
#include "SamplesEnhance.h"
//using namespace sample;
/*get rect center point*/


#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<vector>


using namespace std;
using namespace cv;
using namespace dnn;

vector<string> classes;

vector<String>  SamplesEnhance::getOutputsNames(Net&net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}
void  SamplesEnhance::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.5f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}
void  SamplesEnhance::postprocess(Mat& frame, const vector<Mat>& outs, float confThreshold, float nmsThreshold)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

int  SamplesEnhance::dnnTest()
{
	//string names_file = "/home/oliver/darknet-master/data/coco.names";
	//String model_def = "/home/oliver/darknet-master/cfg/yolov3.cfg";
	//String weights = "/home/oliver/darknet-master/yolov3.weights";

	string names_file = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\data\\myvoc2.names";
	String model_def = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\cfg\\myyolov3-tiny.cfg";
	String weights = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\backup2\\myyolov3-tiny_last.weights";

	int in_w, in_h;
	double thresh = 0.5;
	double nms_thresh = 0.25;
	in_w = in_h = 608;

	//string img_path = "/home/oliver/darknet/data/dog.jpg";
	string img_path = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\scripts\\VOCdevkit\\VOC2014\\JPEGImages\\18.jpg";

	//read names

	ifstream ifs(names_file.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	//init model
	Net net = readNetFromDarknet(model_def, weights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	//read image and forward
	VideoCapture capture(2);// VideoCapture:OENCV中新增的类，捕获视频并显示出来
	while (1)
	{
		Mat frame, blob;
		capture >> frame;


		blobFromImage(frame, blob, 1 / 255.0, Size(in_w, in_h), Scalar(), true, false);

		vector<Mat> mat_blob;
		imagesFromBlob(blob, mat_blob);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		postprocess(frame, outs, thresh, nms_thresh);

		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		imshow("res", frame);

		waitKey(10);
	}
	return 0;
}





Point SamplesEnhance::GetRectCenterPoint(Rect rect)
{
	Point cpt;
	cpt.x = rect.x + cvRound(rect.width / 2.0);
	cpt.y = rect.y + cvRound(rect.height / 2.0);
	return cpt;
}


Rect SamplesEnhance::autoLableforColor(Mat &imgO)
{
		//Mat img = imgO.clone();
		Mat img = imgO;
		Mat imgGray;
		Mat rangemat;
		Mat imgHSV;
		//img=gamaPROC(img);
		const int g_nkernelSize = 5;
		//medianBlur(img, img, g_nkernelSize);
		GaussianBlur(img, img, Size(g_nkernelSize, g_nkernelSize), 2);
		//blur(img, img, Size(g_nkernelSize, g_nkernelSize));
		// img=gamaPROC(img);
		//gray  this is not used in this method
		cvtColor(img, imgGray, CV_BGR2GRAY);

		cvtColor(img, imgHSV, COLOR_BGR2HSV);//杞负HSV
		Mat imgThresholded;
		//split the gray
		//	inRange(imgHSV, Scalar(grayLowH, grayLowS, grayLowV), Scalar(grayHighH, grayHighS, grayHighV), imgThresholded); //Threshold the image

		//erode


		//belt
		int beltLowH = 0;
		int beltHighH = 15;
		int beltLowS = 43;
		int beltHighS = 255;
		int beltLowV = 46;
		int beltHighV = 255;
		//find green and blue  the effect is low
		inRange(imgHSV, Scalar(beltLowH, beltLowS, beltLowV), Scalar(beltHighH, beltHighS, beltHighV), imgThresholded); //Threshold the image
		Mat eromat;
		//   erode(imgThresholded,eromat,element_13); //if not erode the box will tight near the bottle but will generate noise
		eromat = imgThresholded;

		//imshow("imgThresholded", eromat);

		morphologyEx(eromat, eromat, MORPH_OPEN, element_13);

		//draw contour
		vector< vector<Point> > contours;
		//findContours(eromat, contours,CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
		findContours(eromat, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		//findContours(eromat, contours, RETR_FLOODFILL , CV_CHAIN_APPROX_NONE);
		//drawContours(imgO, contours,-1,CV_RGB(0,255,255),2);  //with a thickness of 2
		drawContours(img, contours, -1, CV_RGB(0, 255, 255), 2);  //with a thickness of 2
		int cnt = 0;
		double xmin, ymin, xmax, ymax;
		int theMostleft=1280;
		int theMostup = 960;
		for (int i = 0; i < contours.size(); i++)
		{
			Rect rect = boundingRect(contours[i]);
			Point ct = GetRectCenterPoint(rect);
			double l = arcLength(contours[i], true);
			//if it is a rect the area is enough
			if ((abs(rect.width - rect.height) >(rect.width + rect.height) / 4.0) && (rect.width*rect.height > imgO.size().width*imgO.size().height / 30.0))
			{
				if (rect.x < theMostleft)
					theMostleft = rect.x;
				if (rect.y < theMostup)
					theMostup = rect.y;
				//////rectangle(imgO, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), Scalar(0, 0, 255), 2, 8);
				xmin = rect.x;
				xmax = rect.x + rect.width;
				ymin = rect.y;
				ymax = rect.y + rect.height;

			/////////	cout << "xmin=" << xmin << endl;
			////////	cout << "xmax=" << xmax << endl;
			///////	cout << "ymin=" << ymin << endl;
			////////	cout << "ymax=" << ymax << endl;
				cnt++;
			}

			if (rect.x < theMostleft)
				theMostleft = rect.x;
			if (rect.y < theMostup)
				theMostup = rect.y;

			/////cout << "i" << l << endl;
		}
		/////rectangle(imgO, Point(theMostleft, theMostup), Point(theMostleft + 189, theMostup + 56), Scalar(0, 0, 255), 2, 8);
		//////OutputLabelTXT(imgO, xmin, ymin, xmax, ymax, files_value[i], lable);
		//Label it
		if (cnt == 1)
		{
			/////////	OutputLabelTXT(imgO, xmin, ymin, \
							xmax, ymax, files_value[i], lable);

		}
		//////cout << "contoursize = " << contours.size() << endl;
		//test the gray

#ifdef SHOW
		//namedWindow( "org", CV_WINDOW_NORMAL);
		//resizeWindow( "org", 600, 400);
#endif
		// namedWindow("erod",CV_WINDOW_NORMAL);
		//display the orig pic
#ifdef SHOW
		//imshow( "org", imgO);
		//imshow("process", img);
#endif
		//return img.clone();
	
		Rect resrect(theMostleft, theMostup, theMostleft + nongfu_width, theMostup+nongfu_height);
		return resrect;
	
}

//see python
void RGBtoHSV(int R,int G,int B)
{

}



/*********autolable********/
void SamplesEnhance::autoLableforColorTest(string PICDIR)
{
	vector<string> files_value = listFileFromDIR(PICDIR);
	cout << files_value.size() << endl;
	for (int i = 0; i < files_value.size(); i++)
	{
		cout << files_value[i] << endl;
		String currentdir = (files_value[i]).c_str();
		Mat imgO = imread(currentdir, 1);

		Mat s_down0, s_down1;;
		//decrese the pixel   decrese is not used in this method
		pyrDown(imgO, s_down0, Size(imgO.cols / 2, imgO.rows / 2));

		resize(imgO, s_down1, cv::Size(resize_width, resize_height), (0, 0), (0, 0), cv::INTER_LINEAR);

		//imwrite("/home/hujie1/2_down.jpg",s_down0);
		Mat img = imgO.clone();
		Mat imgGray;
		Mat rangemat;
		Mat imgHSV;
		//img=gamaPROC(img);
		const int g_nkernelSize = 3;
		//medianBlur(img, img, g_nkernelSize);
		//GaussianBlur(img, img, Size(g_nkernelSize, g_nkernelSize), 2);
		//blur(img, img, Size(g_nkernelSize, g_nkernelSize));
		// img=gamaPROC(img);
		//gray  this is not used in this method
		cvtColor(img, imgGray, CV_BGR2GRAY);

		cvtColor(img, imgHSV, COLOR_BGR2HSV);//杞负HSV
		Mat imgThresholded;
		//split the gray
	//	inRange(imgHSV, Scalar(grayLowH, grayLowS, grayLowV), Scalar(grayHighH, grayHighS, grayHighV), imgThresholded); //Threshold the image

																														//erode

		inRange(imgHSV, Scalar(greenLowH, greenLowS, greenLowV), Scalar(greenHighH, greenHighS, greenHighV), imgThresholded); //Threshold the image
		Mat eromat;
		//   erode(imgThresholded,eromat,element_13); //if not erode the box will tight near the bottle but will generate noise
		eromat = imgThresholded;

		morphologyEx(eromat, eromat, MORPH_OPEN, element_13);

		//draw contour
		vector< vector<Point> > contours;
		//findContours(eromat, contours,CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
		findContours(eromat, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		//findContours(eromat, contours, RETR_FLOODFILL , CV_CHAIN_APPROX_NONE);
		//drawContours(imgO, contours,-1,CV_RGB(0,255,255),2);  //with a thickness of 2
		drawContours(img, contours, -1, CV_RGB(0, 255, 255), 2);  //with a thickness of 2
		int cnt = 0;
		double xmin, ymin, xmax, ymax;
		for (int i = 0; i < contours.size(); i++)
		{
			Rect rect = boundingRect(contours[i]);
			Point ct = GetRectCenterPoint(rect);
			double l = arcLength(contours[i], true);
			if ((abs(rect.width - rect.height) > (rect.width + rect.height) / 4.0) && (rect.width*rect.height > imgO.size().width*imgO.size().height / 30.0))
			{
				rectangle(imgO, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), Scalar(0, 0, 255), 2, 8);
				xmin = rect.x;
				xmax = rect.x + rect.width;
				ymin = rect.y;
				ymax = rect.y + rect.height;

				cout << "xmin=" << xmin << endl;
				cout << "xmax=" << xmax << endl;
				cout << "ymin=" << ymin << endl;
				cout << "ymax=" << ymax << endl;
				cnt++;
			}

			cout << "i" << l << endl;
		}
//Label it
		if (cnt == 1)
		{
		/////////	OutputLabelTXT(imgO, xmin, ymin, \
				xmax, ymax, files_value[i], lable);

		}
		cout << "contoursize = " << contours.size() << endl;
		//test the gray

		GaussianBlur(imgGray, imgGray, Size(5, 5), 2);
		inRange(imgGray, 194, 203, rangemat);
		Mat morphmat;
		morphologyEx(rangemat, morphmat, MORPH_OPEN, element_5);
		//Canny(imgGray,imgGray,80,150);
		// namedWindow("bottle",CV_WINDOW_NORMAL);
		// imshow("bottle",rangemat);
		// namedWindow("mo",CV_WINDOW_NORMAL);
		//namedWindow(currentdir+"gray_color",CV_WINDOW_NORMAL);
#ifdef SHOW
		namedWindow(currentdir + "org", CV_WINDOW_NORMAL);
		resizeWindow(currentdir + "org", 600, 400);
#endif
		// namedWindow("erod",CV_WINDOW_NORMAL);
		//display the orig pic
#ifdef SHOW
		imshow(currentdir + "org", imgO);
		imshow(currentdir + "process", img);
#endif
		// imshow("erod",eromat);
		// imshow("mo",morphmat);
		//imshow(currentdir+"gray_color",imgThresholded);
	}
//#ifdef SHOW
//	waitKey(0);
//#endif	
}

/*gama enhance*/
Mat SamplesEnhance::gamaPROC(Mat img)
{//gamma次方 小于1变亮 大于1变暗淡
	//imshow("原始图像", img);
	Mat imgGamma(img.size(), CV_32FC3);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			imgGamma.at<Vec3f>(i, j)[0] = (img.at<Vec3b>(i, j)[0])*(img.at<Vec3b>(i, j)[0])*(img.at<Vec3b>(i, j)[0]);
			imgGamma.at<Vec3f>(i, j)[1] = (img.at<Vec3b>(i, j)[1])*(img.at<Vec3b>(i, j)[1])*(img.at<Vec3b>(i, j)[1]);
			imgGamma.at<Vec3f>(i, j)[2] = (img.at<Vec3b>(i, j)[2])* (img.at<Vec3b>(i, j)[2])* (img.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化
	normalize(imgGamma, imgGamma, 0, 255, CV_MINMAX);

	//转换为8 bit图像显示
	convertScaleAbs(imgGamma, imgGamma);
	//imshow("伽马增强效果", imgGamma);
	return imgGamma;
}


void SamplesEnhance::gamaTest(string PICDIR)
{
	vector<string> files_value = listFileFromDIR(PICDIR);
	vector<string> picdir;
	string dd;
	for (int i = 0; i < files_value.size(); i++)
	{
		String currentdir = (files_value[i]).c_str();
		Mat imgO = imread(currentdir, 1);
		Mat  s_down1;
		s_down1 = gamaPROC(imgO);
		//dd = renameJPG(files_value[i]);
		cout << dd << endl;
		imwrite(currentdir, s_down1);
		//cout << picdir[i] << endl;
	}
	
}


void SamplesEnhance::resizePicTest(string PICDIR, int resize_width, int resize_height)
{
	vector<string> files_value = listFileFromDIR(PICDIR);
	vector<string> picdir;
	string dd;
	for (int i = 0; i < files_value.size(); i++)
	{
		String currentdir = (files_value[i]).c_str();
		Mat imgO = imread(currentdir, 1);
		Mat  s_down1;
		resize(imgO, s_down1, cv::Size(resize_width, resize_height), (0, 0), (0, 0), cv::INTER_LINEAR);
		dd = renameJPG(files_value[i]);
		cout << dd << endl;
		imwrite(dd.c_str(), s_down1);
		//cout << picdir[i] << endl;
	}
}

void SamplesEnhance:: salt(Mat& image, int n) 
{
	for (int k = 0; k<n; k++) {
		int i = rand() % image.cols;
		int j = rand() % image.rows;

		if (image.channels() == 1) {   //判断是一个通道
			image.at<uchar>(j, i) = 255;
		}
		else {
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
}

void SamplesEnhance::saltTest(string PICDIR,int noise_point_num)
{
	vector<string> files_value = listFileFromDIR(PICDIR);
	vector<string> picdir;
	string dd;
	for (int i = 0; i < files_value.size(); i++)
	{
		String currentdir = (files_value[i]).c_str();
		Mat imgO = imread(currentdir, 1);
		Mat  s_down1;
		salt(imgO, noise_point_num);
		imwrite(currentdir, imgO);
		//cout << picdir[i] << endl;
	}
}

Mat SamplesEnhance::mysobel(Mat img_pre)
{
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	Mat grad;
	Sobel(img_pre, grad_x, img_pre.depth(), 1, 0, 3);
	Sobel(img_pre, grad_y, img_pre.depth(), 1, 0, 3);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	return grad;
}


Mat SamplesEnhance::sharp_laplace(Mat input)
{
	Mat h1_kernel = (Mat_<char>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
	Mat h2_kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

	Mat h1_result, h2_result;
	filter2D(input, h1_result, CV_32F, h1_kernel);
	filter2D(input, h2_result, CV_32F, h2_kernel);
	convertScaleAbs(h1_result, h1_result);//abs
	convertScaleAbs(h2_result, h2_result);
	
	//imshow("h1_result", h1_result);  //h2 轻度  h1 重度
	//imshow("h2_result", h2_result);
	return h2_result;
}



void SamplesEnhance::frameSub(Mat img_pre,Mat img_later)
{
	Mat img_pre_gray, img_later_gray;

	const int g_nkernelSize = 25;
	Mat img_pre_med, img_later_med;
	Mat img_pre_blur, img_pre_gauss;
	imshow("orgpre", img_pre);
	//Mat img_pre_sobel= img_pre.clone();
	
	medianBlur(img_pre, img_pre_med, g_nkernelSize);
	medianBlur(img_later, img_later_med, g_nkernelSize);

	blur(img_pre, img_pre_blur, Size(g_nkernelSize, g_nkernelSize));
	GaussianBlur(img_pre, img_pre_gauss, Size(g_nkernelSize, g_nkernelSize),3,3);
	
	Mat img_pre_sharp, img_later_sharp;
	img_pre_sharp=sharp_laplace(img_pre);
	img_later_sharp= sharp_laplace(img_later);

#ifdef SHOW
	//namedWindow("orgpre", CV_WINDOW_NORMAL);
	//resizeWindow("sub", 600, 400);
	imshow("orgpre_med", img_pre_med);
	imshow("orgpre_blur", img_pre_blur);
	imshow("orgpre_gauss", img_pre_gauss);
	imshow("img_pre_sharp", img_pre_sharp);
#endif

	//medianBlur(img_later_gray, img_later_gray, g_nkernelSize);

	//blur(img_pre_gray, img_pre_gray, Size(g_nkernelSize, g_nkernelSize));
	//blur(img_later_gray, img_later_gray, Size(g_nkernelSize, g_nkernelSize));

#ifdef SHOW
	//namedWindow("orglat", CV_WINDOW_NORMAL);
	//resizeWindow("sub", 600, 400);
	imshow("orglat", img_later);
#endif

	//cvtColor(img_pre_sharp, img_pre_gray, CV_BGR2GRAY);
	//cvtColor(img_later_sharp, img_later_gray, CV_BGR2GRAY);

	cvtColor(img_pre_med, img_pre_gray, CV_BGR2GRAY);
	cvtColor(img_later_med, img_later_gray, CV_BGR2GRAY);

	//cvtColor(img_pre, img_pre_gray, CV_BGR2GRAY);
	//cvtColor(img_later, img_later_gray, CV_BGR2GRAY);
	Mat preforeground,bina;
	absdiff(img_pre_gray, img_later_gray, preforeground);

#ifdef SHOW
	//namedWindow("sub", CV_WINDOW_NORMAL);
	//resizeWindow("sub", 600, 400);
	imshow("sub", preforeground);
#endif
	threshold(preforeground, bina, 15, 255,CV_THRESH_BINARY);

	Mat output;
	morphologyEx(bina, output, MORPH_OPEN, element_5);  //先去噪
	morphologyEx(output, output, MORPH_CLOSE, element_13);  //可以的
	//"hecheng"
	//dilate(output, output, element_5);
#ifdef SHOW
	//namedWindow("BINA", CV_WINDOW_NORMAL);
	//resizeWindow("BINA", 600, 400);
	imshow("BINA", bina);
	imshow("morph", output);
#endif


}

void SamplesEnhance::frameSubTest(string PICDIR)
{
	vector<string> files_value = listFileFromDIR(PICDIR);
	cout << files_value[0] << endl;
	//string dir = getDir(files_value[0]);
	Mat imgpre = imread((files_value[0]).c_str(), 1);
	Mat imglater = imread((files_value[1]).c_str(), 1);
	//cout<<imglater.type()<<endl;   16。。。
	frameSub(imgpre, imglater);
}
	/*
	vector<string> picdir;
	string dd;
	for (int i = 0; i < files_value.size(); i++)
	{
		String currentdir = (files_value[i]).c_str();
		Mat imgpre = imread(currentdir, 1);
		

	}
	*/

void SamplesEnhance::AllocateImages(Mat I)
{
	Size sz = I.size();
	int type=I.type();
	IavgF = Mat::zeros(sz, CV_32FC3);
	IdiffF = Mat::zeros(sz, CV_32FC3);
	IprevF = Mat::zeros(sz, CV_32FC3);
	IhiF = Mat::zeros(sz, CV_32FC3);
	IlowF = Mat::zeros(sz, CV_32FC3);

	Ihi1 = Mat::zeros(sz, CV_32FC1);
	Ihi2 = Mat::zeros(sz, CV_32FC1);
	Ihi3 = Mat::zeros(sz, CV_32FC1);
	Ilow1= Mat::zeros(sz, CV_32FC1);
	Ilow2 = Mat::zeros(sz, CV_32FC1);
	Ilow3 = Mat::zeros(sz, CV_32FC1);
	Igray1 = Mat::zeros(sz, CV_32FC1);
	Igray2 = Mat::zeros(sz, CV_32FC1);
	Igray3 = Mat::zeros(sz, CV_32FC1);

	Icount = 0.00001;
	Iscratch = Mat::zeros(sz, CV_32FC3);
	Iscratch2 = Mat::zeros(sz, CV_32FC3);
	//Imaskt= Mat::zeros(sz, CV_8UC1);
	Imaskt = Mat::zeros(sz, CV_32FC1);

}

//Learn the background staticstics for one more frame
//I is a color sample of the background,3-channel
void SamplesEnhance::avgBackground(Mat I)
{
	static int first = 1;
	static int i=0;
	i++;
	//I.convertTo(Iscratch,CV_32FC3,1/255.0);  //除以255则能显示Iscratch 不除则显示不正确
	I.convertTo(Iscratch,CV_32FC3);
	//cout << I.type() << endl;   CV_8UC3
	//cout << CV_32FC3 << endl;
	//cout << CV_8UC3 << endl;
	
	//I.convertTo(Iscratch, CV_32FC3);
	
	//convertScaleAbs(I,Iscratch,1,0);
	//imshow(to_string(i).c_str(), I);
#ifdef SHOW
	imshow((to_string(i) + "I").c_str(), I);
	imshow((to_string(i)+"S").c_str(), Iscratch);
#endif
	//imshow((to_string(i) + "F").c_str(), IavgF);
	cout <<"Iscratch"<< Iscratch.at<Vec3f>(0, 0)[2] << endl;
	cout <<"IavgF_value"<< IavgF.at<Vec3f>(0, 0)[2] << endl;
	cout << "I"<<I.at<Vec3b>(0, 0)[2] << endl;

	cout << "IavgFk:" << IavgF.type() << endl;   //CV_32F 5
	cout << "IscratchK:" << IavgF.type() << endl;   //CV_32F 5
	//waitKey(0);
	if (!first)
	{
		accumulate(Iscratch, IavgF);
		//add(Iscratch, IavgF, IavgF);
		absdiff(IprevF, Iscratch, Iscratch2);
		accumulate(Iscratch2, IdiffF);
		//add(Iscratch, IdiffF, IdiffF);
		Icount += 1.0;
	}
	first = 0;
	IprevF = Iscratch.clone();
}

void SamplesEnhance::setHighThreshold(float scale)
{
	//convertScaleAbs(IdiffF,Iscratch,scale);
	IdiffF.convertTo(Iscratch, CV_32FC3, scale);
	add(Iscratch, IavgF, IhiF);
	Mat IhiArray[3];
#ifdef SHOW
	imshow("hith",IhiF);
#endif
	split(IhiF, IhiArray);
	Ihi1 = IhiArray[0];
	Ihi2= IhiArray[1];
	Ihi3=IhiArray[2];
}

void SamplesEnhance::setLowThreshold(float scale)
{
	//convertScaleAbs(IdiffF, Iscratch, scale);
	IdiffF.convertTo(Iscratch, CV_32FC3, scale);
	//absdiff(IavgF,Iscratch, IlowF);
	subtract(IavgF, Iscratch, IlowF);
#ifdef SHOW
	imshow("lowth", IlowF);
#endif
	Mat IhiArray[3];
	cout << "IavgF:" << IavgF.type() << endl;   //CV_32F 5
	cout << "Iscratch:" << Iscratch.type() << endl;   //CV_32F 5
	cout << "IlowF:" << IlowF.type() << endl;   //CV_32F 5
	split(IlowF, IhiArray);
	Ilow1 = IhiArray[0];
	Ilow2 = IhiArray[1];
	Ilow3 = IhiArray[2];
}


void SamplesEnhance::createModelsfromStats()
{
	cout << "IavgF_pre:" << IavgF.type() << endl;   //CV_32FC3
	//convertScaleAbs(IavgF,IavgF,(double)(1.0/Icount));  //this will change the type to 8u
	//convertScaleAbs(IdiffF, IdiffF, (double)(1.0 / Icount));
	Icount += 1;
	IavgF.convertTo(IavgF, CV_32FC3, (double)(1.0 / Icount));
	IdiffF.convertTo(IdiffF, CV_32FC3, (double)(1.0 / Icount));

	cout << "IavgF_later:" << IavgF.type() << endl;   //CV_8UC3
	add(IdiffF,Scalar(1.0,1.0,1.0),IdiffF);  //这个必须有
	cout << "count" << Icount << endl;
#ifdef SHOW
	imshow("avg", IavgF);
	imshow("diff", IdiffF);
#endif
	cout <<"final"<< IavgF.at<Vec3f>(0, 0)[2] << endl;
	//waitKey(0);
	//setHighThreshold(7.0);
	//setLowThreshold(6.0);

	setHighThreshold(high_threadhold);
	setLowThreshold(low_threadhold);
	//waitKey(0);

}

void SamplesEnhance::backgroundDiff(Mat I,Mat &Imask )
{
	Mat Iscratch_mask;
	Mat Iscratch_mask2;
	Mat Iscratch_mask3;

	Iscratch_mask = Mat::zeros(I.size(), CV_32FC1);
	Iscratch_mask2 = Mat::zeros(I.size(), CV_32FC1);
	Iscratch_mask3 = Mat::zeros(I.size(), CV_32FC1);

#ifdef SHOW
	imshow("inputI", I);
#endif
	//waitKey(0);
	I.convertTo(Iscratch, CV_32FC3);
	//Imask.convertTo(Iscratch_mask, CV_32FC3);
	//Imask.convertTo(Iscratch_mask, Imask.type());
	cout << "type"<<Iscratch_mask.type() << endl;
	//convertScaleAbs(I,Iscratch,1,0);
	Mat IhiArray[3];
	split(Iscratch, IhiArray);
	Igray1 = IhiArray[0];   
	Igray2 = IhiArray[1];
	Igray3 = IhiArray[2];

#ifdef SHOW
	imshow("inputIscratch", Iscratch);
	imshow("Igray1", Igray1);
	imshow("Igray2", Igray2);
	imshow("Igray3", Igray3);

	//waitKey(0);
	//imshow("1", Igray1);
	//imshow("2", Igray2);
	//imshow("3", Igray3);
	imshow("low", Ilow1);
	//imshow("h", Ihi1);
#endif

	cout << "Igray1:" << Igray1.type() << endl;   //CV_32F 5
	cout << "Ilow1:" << Ilow1.type() << endl;   //
	cout << "Ihi1:" << Ihi1.type() << endl;

	//waitKey(0);
	inRange(Igray1, Ilow1, Ihi1, Iscratch_mask);  //will change the type to CV8U

	cout << "Igray1_value" << Igray1.at<Vec3f>(0, 0) << endl;
	cout << "Ilow1_value" << Ilow1.at<Vec3f>(0, 0) << endl;
	cout << "Ihi1_value" << Ihi1.at<Vec3f>(0, 0) << endl;

	cout << "IavgF_value" << IavgF.at<Vec3f>(0, 0) << endl;
	

#ifdef SHOW
	imshow("11", Iscratch_mask);
#endif
	inRange(Igray2, Ilow2, Ihi2, Iscratch_mask2);
	//bitwise_or(Imask, Imaskt, Iscratch_mask);
#ifdef SHOW
	imshow("12", Iscratch_mask2);
#endif
	inRange(Igray3, Ilow3, Ihi3, Iscratch_mask3);
	//bitwise_or(Imask, Imaskt, Iscratch_mask);
#ifdef SHOW
	imshow("13", Iscratch_mask3);
#endif
	cout << "Igray2:" << Igray2.type() << endl;
	cout << "type1:" << Iscratch_mask.type() << endl;
	cout << "type2:" << Iscratch_mask2.type() << endl;

	Mat dst;
	Mat src[3];
	src[0] = Iscratch_mask;
	src[1] = Iscratch_mask2;
	src[2] = Iscratch_mask3;
	//////merge(src,3, dst);

	inRange(Iscratch, IlowF, IhiF, dst);  //will change the type to CV8U

	//subtract(Iscratch_mask,255, Iscratch_mask);
	//imshow("14", Iscratch_mask);
#ifdef SHOW
	imshow("4", dst);
#endif
	//imshow("4", dst);
	//bitwise_not(dst, Imask);
	Imask = dst;
	//waitKey(0);
}
//only support pics same size
void SamplesEnhance::learnBackground(string PICDIR)
{
	vector<string> files_value = listFileFromDIR(PICDIR);
	String currentdir = (files_value[0]).c_str();
	Mat img0 = imread(currentdir, 1);
	AllocateImages(img0);
	for (int i = 0; i < files_value.size(); i++)
	{
		String currentdir = (files_value[i]).c_str();
		Mat img = imread(currentdir, 1);
		avgBackground(img);
	}
	//imshow("avg", IavgF);
	//imshow("diff", IdiffF);
	cout << Icount << endl;
	//waitKey(0);
	createModelsfromStats();
}

void SamplesEnhance::foreGroundSegmentTest(string PICDIR)
{
	//learnBackground(PICDIR);
	learnBackground("E:\\Xscx2019\\vedio\\5");

	vector<string> files_value = listFileFromDIR(PICDIR);
	for (int i = 0; i < files_value.size(); i++)
	{
		String currentdir = (files_value[i]).c_str();
		Mat img = imread(currentdir, 1);
		Mat dst;
		backgroundDiff(img, dst);
		string x = "org";
		imshow(currentdir+x, img);
	    imshow(currentdir, dst);
		//waitKey(0);
	}
}

void SamplesEnhance::LKlightflow_trackTest(string PICDIR)
{
	vector<string> files_value = listFileFromDIR(PICDIR);
	String firstdir = (files_value[0]).c_str();
	Mat featureimg = imread(firstdir, 1);
	for (int i = 1; i < files_value.size(); i++)
	{
		String currentdir = (files_value[i]).c_str();
		Mat secondimg = imread(currentdir, 1);
		Mat dst;
		LKlightflow_track(featureimg, secondimg);
		//waitKey(0);
	}

}


void SamplesEnhance::LKlightflow_trackCamTest()
{
		cvNamedWindow("track", CV_WINDOW_NORMAL);
		//cvResizeWindow("track", 800, 600); //创建一个500*500大小的窗口
		//setWindowProperty("track", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN); //设置窗口全屏
		const int notepad_cam = 0;
		const int usb_cam = 1;
		//VideoCapture captue(notepad_cam);//创建一个摄像头对象并且指定摄像头编号，只有一个写0就可以
		
							   //captue = cvCreateCameraCapture(0);

		string URL="http://192.168.0.93:14345/videostream.cgi?loginuse=admin&loginpas=1";
		//VideoCapture captue(URL.c_str());
		VideoCapture captue(notepad_cam);//创建一个摄像头对象并且指定摄像头编号，只有一个写0就可以

		captue.set(CAP_PROP_FRAME_WIDTH, 640); //设置摄像头采集图像分辨率
		captue.set(CAP_PROP_FRAME_HEIGHT, 480);
		Mat frame;
		captue >> frame;
		cout << frame.size().width << endl;
		cout << frame.size().height << endl;
		Mat pre_frame=Mat::zeros(frame.size(),CV_8UC3);
		pre_frame = frame.clone();
		while (1)
		{
			captue >> frame;
			LKlightflow_track(pre_frame, frame);
			imshow("track", frame);
			//if (waitKey(30)>0)//按下任意键退出摄像头 因电脑环境而异，有的电脑可能会出现一闪而过的情况
			//	break;
			waitKey(10);
			pre_frame = frame.clone();
		}
		captue.release();
		destroyAllWindows();//关闭所有窗口

}

void SamplesEnhance::LKlightflow_track(Mat featureimg, Mat &secondimg_orig)
{
	const int MAX_CORNERS = 500;
	Size img_sz = featureimg.size();
	int win_size = 10;
	Mat imgC;
	
	Mat drawimg = featureimg.clone();
	Mat drawimg2 = secondimg_orig.clone();
	Mat secondimg;
	cvtColor(featureimg, featureimg, CV_BGR2GRAY);
	cvtColor(secondimg_orig, secondimg, CV_BGR2GRAY);
	Mat eig_image = Mat::zeros(img_sz, CV_32FC3);
	Mat tmp_image = Mat::zeros(img_sz, CV_32FC3);
	imgC = Mat::zeros(img_sz, CV_32FC3);
	int corner_count = MAX_CORNERS;
	Mat  cornersA,cornersB;
	goodFeaturesToTrack(featureimg,cornersA, corner_count,0.01,5.0);  // input is grayscale
	////cout << "corner_count" << corner_count << endl;
	cornerSubPix(featureimg, cornersA,Size(win_size,win_size), Size(-1,-1),TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));
///	cout << "width:"<<cornersA.size().width<<endl;  //1
///	cout << "height:" << cornersA.size().height << endl;  //313

///	cout << "value.x" << cornersA.at<Vec2f>(2,0)[0] << endl;  //1   左边是y(height)坐标 右边是x(width)坐标
///	cout << "value.y" << cornersA.at<Vec2f>(2, 0)[1] << endl;  //1   左边是y(height)坐标 右边是x(width)坐标

	int corners_cnt = cornersA.size().height;  //Point(x,y)  x width  y height
	/*
	for (int i = 0;i<corners_cnt;i++)
	{
		Point p0 = Point( cvRound(cornersA.at<Vec2f>(i, 0)[0]), cvRound(cornersA.at<Vec2f>(i, 0)[1]) );
		circle(drawimg, p0, 2, Scalar(0, 0, 255),-1);  //scalar b g r
	}
	*/
#ifdef SHOW
	imshow("corner img", drawimg);
#endif

	//char features_found[MAX_CORNERS];
	//float feature_errors[MAX_CORNERS];
	Mat features_found, feature_errors;
	//Size sz = Size();
	vector<Mat> pyramid1, pyramid2;
	buildOpticalFlowPyramid(featureimg, pyramid1,Size(win_size, win_size), 3);
	buildOpticalFlowPyramid(secondimg, pyramid2, Size(win_size, win_size), 3);
///	cout << "pysize:"<<pyramid1.size()<<endl;
///	cout << "py2size:" << pyramid2.size() << endl;
#ifdef SHOW
	imshow("pyramid1", pyramid1[2]);
	imwrite("E:\\Xscx2019\\vedio\\5\\pyr0.jpg", pyramid1[0]);
	imwrite("E:\\Xscx2019\\vedio\\5\\pyr2.jpg", pyramid1[2]);
	//imshow("pyramid1", pyramid1[2]);
	imshow("pyramid2", pyramid2[2]);
	imwrite("E:\\Xscx2019\\vedio\\5\\p2yr0.jpg", pyramid2[0]);
	imwrite("E:\\Xscx2019\\vedio\\5\\p2yr2.jpg", pyramid2[2]);
	//waitKey(0);
#endif


	//calcOpticalFlowPyrLK(pyramid1[2], pyramid2[2], cornersA, cornersB, features_found, feature_errors);
	calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, cornersB, features_found, feature_errors);

	//cout << "cornersA width:" << cornersA.size().width << endl;   //1
	//cout << "cornersA height:" << cornersA.size().height << endl;//313

	//cout << "features_found width:" << features_found.size().width << endl;  //1
	//cout << "features_found height:" << features_found.size().height << endl;//313

	//cout << "feature_errors width：" << feature_errors.size().width << endl;//1
	//cout << "feature_errors height：" << feature_errors.size().height << endl;//313

	int corners_cntb = cornersB.size().height;
	//cout << "corners_cntb:"<<corners_cntb << endl;
	for (int i = 0; i < corners_cnt; i++)
	{
		float flow_speed = abs(feature_errors.at<float>(i, 0));
		//if (features_found.at<uchar>(i,0) == 0 || feature_errors.at<uchar>(i, 0) > 550)
		if ((features_found.at<uchar>(i, 0) == 0)||(flow_speed < 40)||(flow_speed > 1e5))
		{
			//cout << "filter the error is:" << feature_errors.at<float>(i, 0) << endl;
			continue;
		}
		//cout << "error is:" << feature_errors.at<float>(i, 0) << endl;
		//p0 前一帧的特征角点，可以在yolo里看有没有重合的
		//p1 后一帧的特征角点，可以在yolo里看有没有重合的，对应和cornersA同一个位置的，在前一帧抠出来，
		//又识别到了是同一种东西，证明跟踪到了  将本帧图像的东西和上一帧关联起来  这些角点儿作为媒介

		Point p0 = Point(cvRound(cornersA.at<Vec2f>(i, 0)[0]), cvRound(cornersA.at<Vec2f>(i, 0)[1]));
		Point p1 = Point(cvRound(cornersB.at<Vec2f>(i, 0)[0]), cvRound(cornersB.at<Vec2f>(i, 0)[1]));

		//circle(drawimg2, p1, 2, Scalar(0, 0, 255), -1);  //scalar b g r
		circle(secondimg_orig, p1, 2, Scalar(0, 0, 255), -1);  //scalar b g r
		//line(imgC, p0, p1, Scalar(255, 0, 0),2);
		////line(secondimg_orig, p0, p1, Scalar(0, 255, 255), 1);	
	}
#ifdef SHOW
	//imshow("feature img", featureimg);
	//imshow("second img", secondimg);
	imshow("drawimg2", drawimg2);
	imshow("flow",imgC);
	waitKey(0);
#endif

}
//roughly label the pic at the fixed position,for more precise label on next stage
Mat  SamplesEnhance::roughLabel(Mat input, int left, int top, int right, int down,Rect &outRect)
{
	Mat show = input.clone();
	Rect rect(left, top, abs(left - right), abs(top - down));
	//rectangle(input, rect, Scalar(0, 0, 255), 2, LINE_8, 0);
	Rect rectobj=autoLableforColor(input(rect));
	
	////////OutputLabelTXT(imgO, xmin, ymin, xmax, ymax, files_value[i], lable);
	rectobj.x = rectobj.x + left;
	rectobj.y = rectobj.y + top;

	outRect = rectobj;

	rectangle(show, rectobj, Scalar(0, 0, 255), 2, LINE_8, 0);
	return show;
}

//Mat result = roughLabel(img, 437, 411, 656, 542, outRect);
void SamplesEnhance::roughLabelTest(string PICDIR,int init_left,int init_top,int init_right,int init_down)
{
	int left = 0;
	int top = 0;
	int right = 0;
	int down = 0;
	vector<string> files_value = listFileFromDIR(PICDIR);
	for (int i = 0; i < files_value.size(); i++)
	{

		String currentdir = (files_value[i]).c_str();
		Mat img = imread(currentdir, 1);
		Rect outRect;
		Mat result= roughLabel(img, init_left, init_top, init_right, init_down, outRect);

		int xmin = outRect.x;
		int ymin = outRect.y;
		int xmax = outRect.x+ outRect.width;
		int ymax = outRect.x+ outRect.height;
		int lable = 0;//nong fu shan quan
		OutputLabelTXT_keras(img, xmin, ymin, xmax, ymax, files_value[i], lable);
		//namedWindow(currentdir, 1);
		//resizeWindow(currentdir, Size(480, 320));
		//imshow(currentdir, result);
		imwrite(currentdir, result);
		//waitKey(0);
	}
}






