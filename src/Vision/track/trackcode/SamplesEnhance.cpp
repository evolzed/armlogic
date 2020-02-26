#include "stdafx.h"
#include "SamplesEnhance.h"
//using namespace sample;
/*get rect center point*/
//#include "winSock_client.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<vector>
#include "line_angle.hpp"
#include "winSock_client.h"


//#include "Camera.h"

using namespace std;
using namespace cv;
using namespace dnn;

vector<string> classes;
Mat show;



Point GetCenterPoint(Rect rect)
{
	Point cpt;
	cpt.x = rect.x + cvRound(rect.width / 2.0);
	cpt.y = rect.y + cvRound(rect.height / 2.0);
	return cpt;
}
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
	vector<Rect> boxes;   //ÿ���������и�out��i��  ÿ��out[i]�кܶ���п� 
						//��д����ÿ�����͵Ŀ�λ�ú�Ԥ�����Ŷȣ��ҳ����Ŷ�������

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);  //scores store in  mat
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);  //data ǰ5����x y width height  Ȼ���� confidence �ֳ�ÿ�����confidence
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
		//drawPred(classIds[idx], confidences[idx], box.x, box.y,
		//	box.x + box.width, box.y + box.height, frame);

		////drawPred(classIds[idx], confidences[idx], box.x, box.y,
			/////box.x + box.width, box.y + box.height, show);
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void  SamplesEnhance::studyBackgroundFromCam(camera cam)
{
	string BGDIR = "E:\\Xscx2019\\OPENCV_PROJ\\backgroundLearn\\";
	int over_flag = 1;
	while (over_flag)
	{
		static int pic_cnt = 1;
		Mat frame, blob;
		
		frame = cam.getImage();
		Mat fin = Mat::zeros(frame.size(), CV_32FC3);
			
		frame.convertTo(fin, CV_32FC3);

		bgVector.push_back(fin);
		string picname = to_string(pic_cnt);
		picname = BGDIR + picname + ".jpg";
		cout << picname << endl;
	////imwrite(picname.c_str(), frame);
		waitKey(200);
		pic_cnt++;
		if (pic_cnt == BG_STUDY_NUM)
			over_flag = 0;
	}

	////learnBackground(BGDIR);
	learnBackGroundFromVec(bgVector);
}


void  SamplesEnhance::getPicFromCam(camera cam)
{
	string BGDIR = "E:\\Xscx2019\\OPENCV_PROJ\\GetPic\\";
	int over_flag = 1;
	while (over_flag)
	{
		static int pic_cnt = 1;
		Mat frame, blob;
		frame = cam.getImage();
		string picname = to_string(pic_cnt);
		picname = BGDIR + picname + ".jpg";
		cout << picname << endl;
		imwrite(picname.c_str(), frame);
		waitKey(1000);
		pic_cnt++;
		if (pic_cnt == 120)
			over_flag = 0;
	}
	
}


void SamplesEnhance::loadBackgroundModel()
{
	Mat IlowF0, IhiF0;
	IlowF0 = imread(BG_LOW_THRESHOLD_DIR.c_str());
	IhiF0 = imread(BG_HIGH_THRESHOLD_DIR.c_str());

	//IlowF0 = imread(BG_LOW_THRESHOLD_DIR.c_str(), IMREAD_ANYDEPTH);
	//IhiF0 = imread(BG_HIGH_THRESHOLD_DIR.c_str(), IMREAD_ANYDEPTH);
	
	cout << IlowF0.type() << endl;
	cout << IhiF0.type() << endl;
	imshow("IlowF0", IlowF0);
	imshow("IhiF0", IhiF0);


//	IlowF0.convertTo(IlowF, CV_32FC3,1.0/255.0);
//	IhiF0.convertTo(IhiF, CV_32FC3, 1.0 / 255.0);
	IlowF0.convertTo(IlowF, CV_32FC3);
	IhiF0.convertTo(IhiF, CV_32FC3);

	///IlowF0.convertTo(IlowF, IlowF0.type());
	///IhiF0.convertTo(IhiF, IhiF0.type());

	cout << IlowF.type() << endl;
	cout << IhiF.type() << endl;
	imshow("IlowF", IlowF);
	imshow("IhiF", IhiF);
}

//extern void winSockclientTest(SOCKET sockClient);
//extern void winSockclientTest();
int  SamplesEnhance::dnnTest() 
{
	//string names_file = "/home/oliver/darknet-master/data/coco.names";
	//String model_def = "/home/oliver/darknet-master/cfg/yolov3.cfg";
	//String weights = "/home/oliver/darknet-master/yolov3.weights";
	/////////SOCKET sockClient;


	//winSockclientTest();
	
	winSockclientInit();
	//string img_path = "/home/oliver/darknet/data/dog.jpg";
	string img_path = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\scripts\\VOCdevkit\\VOC2014\\JPEGImages\\18.jpg";

	//read names

	///vector<Mat> bgVector;

	

	camera cam;
	cam.init();
	if(bgNeedUpdated)
		studyBackgroundFromCam(cam);
	else
	{
		loadBackgroundModel();
		//IlowF = imread(BG_LOW_THRESHOLD_DIR.c_str());
		//IhiF = imread(BG_HIGH_THRESHOLD_DIR.c_str());
		//IlowF.convertTo(IlowF, CV_32FC3);
		//IhiF.convertTo(IhiF, CV_32FC3);
	}
/////	loadBackgroundModel();
	/////////getPicFromCam(cam);
	Mat frame;
	frame = cam.getImage();
	/////capture >> frame;
	Mat pre_frame = Mat::zeros(frame.size(), CV_8UC3);
	pre_frame = frame.clone();
	show= frame.clone();

	Mat backMask_init;
	vector<Rect> tmp;
	backgroundDiff(pre_frame, backMask_init, tmp);

	Mat frame_delimite_bac_pre;

	bitwise_and(pre_frame, pre_frame, frame_delimite_bac_pre, backMask_init);



	/*
	while (1)
	{
		capture >> frame;
		LKlightflow_track(pre_frame, frame);
		imshow("track", frame);
		//if (waitKey(30)>0)//����������˳�����ͷ ����Ի������죬�еĵ��Կ��ܻ����һ��
		���������
		//	break;
		waitKey(10);
		pre_frame = frame.clone();
	*/
	

	/******************nn*******************/
	string names_file = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\data\\myvoc2.names";
	String model_def = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\cfg\\myyolov3-tiny.cfg";
	String weights = "E:\\Xscx2019\\OPENCV_PROJ\\darknet-master\\backup2\\myyolov3-tiny_last.weights";

	ifstream ifs(names_file.c_str());
	string line;
	while (getline(ifs, line)) 
		classes.push_back(line);

	int in_w, in_h;
	double thresh = 0.5;
	double nms_thresh = 0.25;
	in_w = in_h = 608;
	////in_w = in_h = 1280;

	//init model
	Net net = readNetFromDarknet(model_def, weights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	//net.setPreferableTarget(DNN_TARGET_OPENCL);

	//read image and forward
	/////////	VideoCapture capture(0);// VideoCapture:OENCV���������࣬������Ƶ����ʾ����
	/******************nn*******************/
	vector<Vec2d> basketLines;
	while (1)
	{
		Mat frame, blob;
		///////capture >> frame;
		frame = cam.getImage();
		
		show = frame.clone();
		
		Mat dst;
		vector<Rect> rectArray;
		backgroundDiff(frame,dst, rectArray);

		string x = " bac";
		//imshow(currentdir + x, img);
		//imshow(x.c_str(), dst);

		/////threshold(dst, dst, 30, 255,0);//same direction 
		//imshow("bina", dst);
		Mat backMask;
		dst.convertTo(backMask, CV_8UC3); //change to C3
		imshow("C3", dst);
		Mat frame_delimite_bac;
		cout << frame.type()<<endl; 
		cout << backMask.type()<<endl;

		//bitwise_and(frame, backMask, frame_delimite_bac);
		bitwise_and(frame, frame,frame_delimite_bac, backMask);

		Rect rectFlag0;
		findrColor(frame,rectFlag0);
		rectFlag = rectFlag0;
		rectFlag.width = (1280 - (rectFlag.x));
		rectFlag.height = (960 - (rectFlag.y));

		Rect rectPool1;
		rectPool1.x = rectFlag.x + 200;
		rectPool1.y = rectFlag.y + 70;
		rectPool1.width =90 ;
		rectPool1.height =50 ;


		////rectangle(show,rectFlag,Scalar(0,255,255));
		rectangle(show, rectPool1, Scalar(0, 255, 255));

		


		Mat gray;
		cvtColor(frame, gray, CV_RGB2GRAY);
		//imshow("gray",gray);
		///HoughLines()
		//Mat theobj=gray(rectFlag).clone();
		double kout, bout;
		if (((rectPool1.x > 10) && (rectPool1.y > 10)) && (rectPool1.x + rectPool1.width + 10 < frame.size().width) && (rectPool1.y + rectPool1.height + 10 < frame.size().height))
		{		//protect
			Mat theobj = gray(rectPool1).clone();
			//Mat theobj1 = frame(rectFlag).clone();
			//theobj1 = gamaPROC(theobj1);
			imshow("theobj", theobj);

			//imshow("theobjgamma", theobj);

			Mat h2_kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

			Mat h2_result;

			////	filter2D(theobj, h2_result, CV_8UC1, h2_kernel);

			///	convertScaleAbs(h2_result, h2_result);


			////	imshow("h2_result", h2_result);



			Canny(theobj, theobj, 10, 50);

			//������ �ҵ�ļ�  fitline

			//Canny(gray(rectFlag), gray(rectFlag), 10, 100);
			//Mat lines;
			vector<Vec4d> Lines;
			//HoughLinesP(theobj, Lines, 1, CV_PI / 360, 5, 5, 35);
			//HoughLinesP(theobj, Lines, 50, 6 * CV_PI / 180, 2, 1, 35);
			HoughLinesP(theobj, Lines, 1, CV_PI / 180, 30, 10, 3);
			imshow("canny", theobj);
			/*
			vector<Vec4d> Lines_convert;
			for (size_t i = 0; i < Lines.size(); i++)
			{

			//	cv::line(show, Point(Lines[i][0], Lines[i][1]) + Point(rectFlag.x, rectFlag.y), Point(Lines[i][2], Lines[i][3]) + Point(rectFlag.x, rectFlag.y), Scalar(0, 0, 255), 2, 8);

				Lines_convert[i][0] = Lines[i][0] + rectFlag.x;
				Lines_convert[i][1] = Lines[i][1] + rectFlag.y;

				Lines_convert[i][2] = Lines[i][2] + rectFlag.x;
				Lines_convert[i][3] = Lines[i][3] + rectFlag.y;
			}
			*/
			vector<Vec4d> Lines_convert;
			if (Lines.size() > 0)
			{
				// Lines_convert = convertRoiCoordinate(Lines, rectFlag);
				Lines_convert = convertRoiCoordinate(Lines, rectPool1);
			}
		
			cout<<"Lines.size()="<< Lines.size() << endl;
		//	double kout, bout;
			for (size_t i = 0; i < Lines.size(); i++)
			{
				//cv::line(show, Point(Lines[i][0], Lines[i][1])+Point(rectFlag.x, rectFlag.y), Point(Lines[i][2], Lines[i][3])+Point(rectFlag.x, rectFlag.y), Scalar(0, 0, 255), 2, 8);
				//cv::line(show, Point(Lines[i][0], Lines[i][1]) + Point(rectPool1.x, rectPool1.y), Point(Lines[i][2], Lines[i][3]) + Point(rectPool1.x, rectPool1.y), Scalar(255, 0, 255), 2, 8);
				Vec4d line = Lines_convert[i];
				double cos_value = cos_value_to_horizon(line);
				double deg = acos(cos_value) / 3.1415926 * 180;
				if ((deg > 70) && (deg < 90))
				{
					double k, b;
					resolveLineKXEqution(line, k, b);
					Vec2d kb;
					kb[0] = k;
					kb[1] = b;

					///basketLines.insert(basketLines.begin(),kb);//insert to head ���²�������
					basketLines.push_back(kb);//insert to head ���²�������
					cout << basketLines.size() << endl;
					if(basketLines.size()==15)
					{
						double sumk=0, sumb=0;
						for (int i = 0;i < basketLines.size();i++)
						{
							sumk += basketLines[i][0];
							sumb += basketLines[i][1];
						}
						kout = sumk / 15.0;
						bout = sumb / 15.0;
						basketLines.erase(basketLines.begin());  //del the end
					}

					//cv::line(show, Point(Lines_convert[i][0], Lines_convert[i][1]), Point(Lines_convert[i][2], Lines_convert[i][3]), Scalar(0, 0, 255), 2, 8);
					//cv::line(show, Point(Lines_convert[i][0], Lines_convert[i][1]), Point(xV, yV), Scalar(0, 255, 255), 2, 8);
					
					
				}
			}
			//solution ins
			//double xV = -bout / kout; 
			double xV = solutionXlineKXEqution(kout, bout, 0);
			double yV = 0;

			//double xb = (960 - bout) / kout;  
			double xb = solutionXlineKXEqution(kout, bout, 960);
			double yb = (960);
			cv::line(show, Point(xb,yb), Point(xV, yV), Scalar(0, 255, 255), 1, 8);


		}



		//SOCKET sockClient;
		if (rectArray.size() > 0)
		{
			for (int i = 0;i < rectArray.size();i++)
			{
				Point ct=GetCenterPoint(rectArray[i]);
				circle(show, ct, 6, Scalar(0, 0, 255), -1);
				double xlimit= solutionXlineKXEqution(kout, bout, ct.y);
				circle(show, Point(xlimit, ct.y), 9, Scalar(0, 100, 255), 1);
				if( ((xlimit-ct.x)<100) && ((xlimit - ct.x)>0) && (ct.y<680)&&(ct.y>260)  )
				{
					socksend();
				}

			}
		}
		
	
				/*
				if ((deg > 50) && (deg < 90))
				{

					cv::line(show, Point(Lines_convert[i][0], Lines_convert[i][1]), Point(Lines_convert[i][2], Lines_convert[i][3]), Scalar(0, 0, 255), 2, 8);
					basketLines.push_back(Lines_convert[i]);
				}
				*/
				//cout << "deg="<<acos(cos_value) << endl;
				//cout << "cos=" << cos_value << endl;
				//cout << "deg=" << acos(cos_value)/3.1415926*180 << endl;

			
			//find the different lines
			/*
			vector<Vec4d> basketLinesChoose;
			cout << "line num" << basketLines.size() << endl;
			vector<int> indexForDel;
			if (basketLines.size() > 0)
			{
				//del the distance small
				//static int quitFlagCnt = 0;
				for (size_t j = 0; j < basketLines.size(); j++)
				{

					for (size_t i = j + 1; i < basketLines.size(); i++)
					{
						Vec4d line = basketLines[i];
						double distance = distance_between_lines(basketLines[j], basketLines[i]);
						if (distance < 20)
						{
							indexForDel.push_back(j);
						}

					}
				}

				for (size_t i = 0; i < basketLines.size(); i++)
				{
					bool notEqual = false;
					for (size_t j = 0; j < indexForDel.size(); j++)
					{
						if (i == indexForDel[j])
							notEqual = true;


					}
					if (notEqual == false)  //illustarte that i should be retain
						basketLinesChoose.push_back(basketLines[i]);
				}




			}
			*/



			/*
			if ((distance > 220) && (distance < 260))
			{
			basketLinesChoose.push_back(basketLines[i]);
			basketLinesChoose.push_back(basketLines[i-1]);

			}
			

			if (basketLinesChoose.size() > 0)
			{
				for (size_t i = 1; i < basketLinesChoose.size(); i++)
				{
					Vec4d line = basketLinesChoose[i];


					cv::line(show, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), 2, 8);
					cout << "frameNUM=" << cam.frameNum;
				}
			}
			*/
		

		imshow("and", frame_delimite_bac);

		////LKlightflow_track(pre_frame, frame);   //track the current frame


	////	LKlightflow_track(frame_delimite_bac_pre, frame_delimite_bac);   //track the current frame of back delete

	////	imshow("and", frame_delimite_bac);//see track 
		////blobFromImage(frame, blob, 1 / 255.0, Size(in_w, in_h), Scalar(), true, false);


/******************nn*******************
		blobFromImage(frame_delimite_bac, blob, 1 / 255.0, Size(in_w, in_h), Scalar(), true, false);

		vector<Mat> mat_blob;
		imagesFromBlob(blob, mat_blob);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		/////postprocess(frame, outs, thresh, nms_thresh);  //draw

		postprocess(frame_delimite_bac, outs, thresh, nms_thresh);  //draw

		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		//putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		////putText(show, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
		putText(frame_delimite_bac, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
*************************************/

		pre_frame = frame.clone();

		frame_delimite_bac_pre = frame_delimite_bac.clone();
		///imshow("res", frame);
		///cvNamedWindow("res", CV_WINDOW_FULLSCREEN);
		imshow("res", show);
	//	imshow("res2", frame_delimite_bac);
		if (waitKey(10) == 27) //��ʱ30ms,�����������ʲ�����Ƶ,�����ڼ䰴�����ⰴ�����˳���Ƶ���ţ������ؼ�ֵ
			break;
		//
		waitKey(10);
	}
	cam.close();

	////closesocket(sockClient);
	///WSACleanup();


	sockclose();
	return 0;
}





Point SamplesEnhance::GetRectCenterPoint(Rect rect)
{
	Point cpt;
	cpt.x = rect.x + cvRound(rect.width / 2.0);
	cpt.y = rect.y + cvRound(rect.height / 2.0);
	return cpt;
}


//only for xueb based on color
Rect SamplesEnhance::autoLableforXueB(Mat &imgO)
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

	cvtColor(img, imgHSV, COLOR_BGR2HSV);//转为HSV
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
	//inRange(imgHSV, Scalar(beltLowH, beltLowS, beltLowV), Scalar(beltHighH, beltHighS, beltHighV), imgThresholded); //Threshold the image
	///inRange(imgHSV, Scalar(greenLowH, greenLowS, greenLowV), Scalar(greenHighH, greenHighS, greenHighV), imgThresholded); //Threshold the image

	inRange(imgHSV, Scalar(yellowLowH, yellowLowS, yellowLowV), Scalar(yellowHighH, yellowHighS, yellowHighV), imgThresholded); //Threshold the image
	Mat eromat;
	//   erode(imgThresholded,eromat,element_13); //if not erode the box will tight near the bottle but will generate noise
	eromat = imgThresholded;

	//imshow("imgThresholded", eromat);
#ifdef SHOW
	imshow("color_split", eromat);
#endif
	//morphologyEx(eromat, eromat, MORPH_OPEN, element_5);
	dilate(eromat, eromat, element_7);
	//dilate(eromat, eromat, element_5);
#ifdef SHOW
	imshow("dialte", eromat);
#endif
	//draw contour
	vector< vector<Point> > contours;
	//findContours(eromat, contours,CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
	findContours(eromat, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//findContours(eromat, contours, RETR_FLOODFILL , CV_CHAIN_APPROX_NONE);
	//drawContours(imgO, contours,-1,CV_RGB(0,255,255),2);  //with a thickness of 2
	cout << contours.size() << endl;
	drawContours(img, contours, -1, CV_RGB(0, 255, 255), 2);  //with a thickness of 2
#ifdef SHOW
	imshow("img_contoure", img);
	//waitKey();
#endif
	int cnt = 0;
	double xmin, ymin, xmax, ymax;
	int theMostleft = 1280;
	int theMostup = 960;
	for (int i = 0; i < contours.size(); i++)
	{
		Rect rect = boundingRect(contours[i]);
		Point ct = GetRectCenterPoint(rect);
		double l = arcLength(contours[i], true);

		if (rect.x < theMostleft)
		{ 
			theMostleft = rect.x;
			theMostup = rect.y;
			theMostup -= 25;
		}
			
		//if (rect.y < theMostup)
			//theMostup = rect.y;
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

	Rect resrect(theMostleft, theMostup, theMostleft + xuebi_width, theMostup + xuebi_height);

	//Rect resrect(xmin, ymin, xmax -xmin, ymax - ymin);
	return resrect;

}

void rectExtend(Rect &rect, int width, int height)
{
	rect.x -= width;
	rect.y -= height;
	rect.width += 2*width;
	rect.height += 2*height;
	//return rect;
}



void SamplesEnhance::findrColor(Mat &imgO,Rect &rectout)
{
	//Mat img = imgO.clone();
	Mat img = imgO;
	Mat imgGray;
	Mat rangemat;
	Mat imgHSV;

	const int g_nkernelSize = 13;
	//medianBlur(img, img, g_nkernelSize);
	GaussianBlur(img, img, Size(g_nkernelSize, g_nkernelSize), 2);
	cvtColor(img, imgGray, COLOR_BGR2GRAY);//转为HSV

	cvtColor(img, imgHSV, COLOR_BGR2HSV);//转为HSV
	Mat imgThresholded;
	

	//find green and blue  the effect is low
	inRange(imgHSV, Scalar(redLowH, redLowS, redLowV), Scalar(redHighH, redHighS, redHighV), imgThresholded); //Threshold the image
	Mat eromat;
	//   erode(imgThresholded,eromat,element_13); //if not erode the box will tight near the bottle but will generate noise
	eromat = imgThresholded;

	//imshow("imgThresholded", eromat);
	imshow("color0", eromat);
	morphologyEx(eromat, eromat, MORPH_OPEN, element_5);
	imshow("color", eromat);



	//draw contour
	vector< vector<Point> > contours;
	//findContours(eromat, contours,CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);
	findContours(eromat, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//findContours(eromat, contours, RETR_FLOODFILL , CV_CHAIN_APPROX_NONE);
	//drawContours(imgO, contours,-1,CV_RGB(0,255,255),2);  //with a thickness of 2
	
	cout << contours.size() << endl;
	int csize = contours.size();
	if (csize > 0)
	{
		int cnt = 0;
		double xmin, ymin, xmax, ymax;
		int theMostleft = 1280;
		int theMostup = 960;
		double area_max = 0;
		int the_index;
		for (int i = 0; i < contours.size(); i++)
		{
			Rect rect = boundingRect(contours[i]);
			Point ct = GetRectCenterPoint(rect);
			double l = arcLength(contours[i], true);

			double area = contourArea(contours[i]);
			if (area > area_max)
			{
				area_max = area;
				the_index = i;
			}

			//rectangle(show, rect, Scalar(255, 0, 0));
		}
		Rect rect = boundingRect(contours[the_index]);

		//rect.x -= 20;
		//rect.x -= 20;
		//rect.width += 20;
		//rect.height += 20;
		if(((rect.x>10)&&(rect.y>10))&& (rect.x+rect.width+10<imgGray.size().width)&&(rect.y+rect.height+10<imgGray.size().height))
			rectExtend(rect, 10, 10);
		std::vector<Vec3f> circles;//�洢ÿ��Բ��λ����Ϣ
								   //����Բ
		//HoughCircles(imgGray(rect), circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);

		Mat theCir = imgGray(rect).clone(); //not put in the for::
		imshow("theCir", theCir);
		Canny(theCir, theCir, 10, 30);
		//HoughCircles(theCir, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);
		HoughCircles(theCir, circles, CV_HOUGH_GRADIENT, 1.5, 20, 100, 100, 0, 0);
		imshow("theCir", theCir);
		//HoughCircles(imgGray(rect), circles, CV_HOUGH_GRADIENT, 1, 10, 200, 100, 0, 0);//will error
		if (circles.size() > 1)
		{
			for (size_t i = 0; i < circles.size(); i++)
			{
				///Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
				Point center(cvRound(circles[i][0])+rect.x, cvRound(circles[i][1]+rect.y));

				int radius = cvRound(circles[i][2]);
				//std::cout << "Բ��X��" << circles[i][0] << "Բ��Y��" << circles[i][1] << std:: endl;
				//����Բ����  
				circle(show, center, radius, Scalar(155, 50, 255), 3, 8, 0);
			}
		}
		rectangle(show, rect, Scalar(255, 0, 0));
		rectout = rect;
		////drawContours(show, contours, -1, CV_RGB(0, 255, 255), 2);  //with a thickness of 2
	}
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

		cvtColor(img, imgHSV, COLOR_BGR2HSV);//转为HSV
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

		cvtColor(img, imgHSV, COLOR_BGR2HSV);//转为HSV
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
{//gamma�η� С��1���� ����1�䰵��
	//imshow("ԭʼͼ��", img);
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
	//��һ��
	normalize(imgGamma, imgGamma, 0, 255, CV_MINMAX);

	//ת��Ϊ8 bitͼ����ʾ
	convertScaleAbs(imgGamma, imgGamma);
	//imshow("٤����ǿЧ��", imgGamma);
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

		if (image.channels() == 1) {   //�ж���һ��ͨ��
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
	
	//imshow("h1_result", h1_result);  //h2 ���  h1 �ض�
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
	morphologyEx(bina, output, MORPH_OPEN, element_5);  //��ȥ��
	morphologyEx(output, output, MORPH_CLOSE, element_13);  //���Ե�
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
	//cout<<imglater.type()<<endl;   16������
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
	//I.convertTo(Iscratch,CV_32FC3,1/255.0);  //����255������ʾIscratch ��������ʾ����ȷ
	I.convertTo(Iscratch,CV_32FC3);
	//cout << I.type() << endl;   CV_8UC3
	//cout << CV_32FC3 << endl;
	//cout << CV_8UC3 << endl;
	cout << "Iscratch"<<Iscratch.type() << endl;  
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
	////imwrite(BG_HIGH_THRESHOLD_DIR.c_str(), IhiF);
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
	//////imwrite(BG_LOW_THRESHOLD_DIR.c_str(), IlowF);
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
	add(IdiffF,Scalar(1.0,1.0,1.0),IdiffF);  //���������
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



int FindOptimumContourForBarcode(vector<std::vector<cv::Point>> contours, Rect &rect_res)
{
	if (contours.size() > 0)
	{
		double max = 0;
		int maxIndex = 0;
		
		vector<Moments> mom(contours.size());
		vector<Point2f> m(contours.size());
		for (int index = 0; index < contours.size(); index++)
		{
			if (contours.size() == 0)
			{
				break;
			}
			double tmp = fabs(contourArea(contours[index]));
			Rect rect = boundingRect(contours[index]);
			Point ct = GetCenterPoint(rect);
			mom[index] = moments(contours[index], false);
			m[index] = Point(static_cast<float>(mom[index].m10 / mom[index].m00), static_cast<float>(mom[index].m01 / mom[index].m00));
			if (abs(ct.x - m[index].x) < rect.width / 10.0)
			{
				if ((double)abs(rect.width - rect.height) < 0.5*(rect.width + rect.height))
				{
					if (tmp >= max)
					{
						max = tmp;
						maxIndex = index;
					}
				}
			}
		}
		Rect rect = boundingRect(contours[maxIndex]);

		if ((abs((double)rect.width / (double)rect.height - 1)) < 0.7)
		{
			rect_res = rect;
			return maxIndex;

		}
		else
			return -1;
	}
	else
		return -1;

}


void SamplesEnhance::backgroundDiff(Mat I,Mat &Idst,vector<Rect> &rectArray )
{
	/*
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
	*/
	Mat dst;
	I.convertTo(Iscratch, CV_32FC3);

	
	///I.convertTo(Iscratch, I.type());
	
	/////cout << Iscratch.type();
	inRange(Iscratch, IlowF, IhiF, dst);  //will change the type to CV8U  only this is useful
	
	//morphologyEx(dst, dst, MORPH_OPEN, element_5);
	//morphologyEx(dst, dst, MORPH_CLOSE, element_5);

	morphologyEx(dst, dst, MORPH_OPEN, element_7);
	morphologyEx(dst, dst, MORPH_CLOSE, element_7);
	
	//threashold()

	//absdiff(dst,255, dst); 
	subtract(255,dst, dst);
	
	
	////morphologyEx(dst, dst, MORPH_OPEN, element_ecllipse_13);
	////medianBlur(dst, dst, 13);
	GaussianBlur(dst, dst, Size(19, 19),3,3);
	dilate(dst, dst, element_ecllipse_19);
	dilate(dst, dst, element_19);
	morphologyEx(dst, dst, MORPH_CLOSE, element_ecllipse_13);


	vector<std::vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(show, contours, -1, Scalar(255, 0, 0));

	if (contours.size() > 0)
	{
		double max = 0;
		int maxIndex = 0;

		vector<Moments> mom(contours.size());
		vector<Point2f> m(contours.size());
		for (int index = 0; index < contours.size(); index++)
		{

			double tmp = fabs(contourArea(contours[index]));
			Rect rect = boundingRect(contours[index]);
			Point ct = GetCenterPoint(rect);//��������
			mom[index] = moments(contours[index], false);
			m[index] = Point(static_cast<float>(mom[index].m10 / mom[index].m00), static_cast<float>(mom[index].m01 / mom[index].m00));//����
			/*
			if (abs(ct.x - m[index].x) < rect.width / 10.0)//�������ĺ������غ�
			{
				if ((double)abs(rect.width - rect.height) < 0.5*(rect.width + rect.height))  //������
				{
					if (tmp >= max)
					{
						max = tmp;
						maxIndex = index;
					}
				}
			}
			*/
			Rect rect0 = boundingRect(contours[index]);
			rectangle(show, rect0 , Scalar(0, 255, 0));
			rectArray.push_back(rect0);
		}
		//////Rect rect = boundingRect(contours[maxIndex]);
	}

	//morphologyEx(dst, dst, MORPH_OPEN, element_ecllipse_13);
	//morphologyEx(dst, dst, MORPH_OPEN, element_ecllipse_13);
	//dilate(dst, dst, element_13);
	//dilate(dst, dst, element_13);
	////dilate(dst, dst, element_ecllipse_13);
	

	//dilate(dst, dst, element_13);
	//subtract(Iscratch_mask,255, Iscratch_mask);
	//imshow("14", Iscratch_mask);
#ifdef SHOW
	imshow("4", dst);
#endif
	//imshow("4", dst);
	//bitwise_not(dst, Imask);
	Idst = dst;
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


void SamplesEnhance::learnBackGroundFromVec(vector<Mat> bgVector)
{
	Mat img0 = bgVector[0];
	AllocateImages(img0);
	for (int i = 0; i < bgVector.size(); i++)
	{
		Mat img = bgVector[i];
		cout << "img.type"<<img.type() << endl;
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
		vector<Rect> tmp;
		backgroundDiff(img, dst,tmp);
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
		//cvResizeWindow("track", 800, 600); //����һ��500*500��С�Ĵ���
		//setWindowProperty("track", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN); //���ô���ȫ��
		const int notepad_cam = 0;
		const int usb_cam = 1;
		//VideoCapture captue(notepad_cam);//����һ������ͷ������ָ������ͷ��ţ�ֻ��һ��д0�Ϳ���
		
							   //captue = cvCreateCameraCapture(0);

		string URL="http://192.168.0.93:14345/videostream.cgi?loginuse=admin&loginpas=1";
		//VideoCapture captue(URL.c_str());
		VideoCapture captue(notepad_cam);//����һ������ͷ������ָ������ͷ��ţ�ֻ��һ��д0�Ϳ���

		captue.set(CAP_PROP_FRAME_WIDTH, 640); //��������ͷ�ɼ�ͼ��ֱ���
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
			//if (waitKey(30)>0)//����������˳�����ͷ ����Ի������죬�еĵ��Կ��ܻ����һ�����������
			//	break;
			waitKey(10);
			pre_frame = frame.clone();
		}
		captue.release();
		destroyAllWindows();//�ر����д���

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

///	cout << "value.x" << cornersA.at<Vec2f>(2,0)[0] << endl;  //1   �����y(height)���� �ұ���x(width)����
///	cout << "value.y" << cornersA.at<Vec2f>(2, 0)[1] << endl;  //1   �����y(height)���� �ұ���x(width)����

	int corners_cnt = cornersA.size().height;  //Point(x,y)  x width  y height
	/*
	for (int i = 0;i<corners_cnt;i++)
	{
		Point p0 = Point( cvRound(cornersA.at<Vec2f>(i, 0)[0]), cvRound(cornersA.at<Vec2f>(i, 0)[1]) );
		circle(drawimg, p0, 2, Scalar(0, 0, 255),-1);  //scalar b g r
	}
	*/
#ifdef SHOW
	////imshow("corner img", drawimg);
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
	//imshow("pyramid1", pyramid1[2]);
	//imwrite("E:\\Xscx2019\\vedio\\5\\pyr0.jpg", pyramid1[0]);
	//imwrite("E:\\Xscx2019\\vedio\\5\\pyr2.jpg", pyramid1[2]);
	//imshow("pyramid1", pyramid1[2]);
	//imshow("pyramid2", pyramid2[2]);
	//imwrite("E:\\Xscx2019\\vedio\\5\\p2yr0.jpg", pyramid2[0]);
	//imwrite("E:\\Xscx2019\\vedio\\5\\p2yr2.jpg", pyramid2[2]);
	//waitKey(0);
#endif


	//calcOpticalFlowPyrLK(pyramid1[2], pyramid2[2], cornersA, cornersB, features_found, feature_errors);
	calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, cornersB, features_found, feature_errors);

	//cout << "cornersA width:" << cornersA.size().width << endl;   //1
	//cout << "cornersA height:" << cornersA.size().height << endl;//313

	//cout << "features_found width:" << features_found.size().width << endl;  //1
	//cout << "features_found height:" << features_found.size().height << endl;//313

	//cout << "feature_errors width��" << feature_errors.size().width << endl;//1
	//cout << "feature_errors height��" << feature_errors.size().height << endl;//313

	int corners_cntb = cornersB.size().height;
	//cout << "corners_cntb:"<<corners_cntb << endl;
	for (int i = 0; i < corners_cnt; i++)
	{
		float flow_speed = abs(feature_errors.at<float>(i, 0));
		//if (features_found.at<uchar>(i,0) == 0 || feature_errors.at<uchar>(i, 0) > 550)
		//if ((features_found.at<uchar>(i, 0) == 0)||(flow_speed < 40)||(flow_speed > 1e5))
		if ((features_found.at<uchar>(i, 0) == 0) || (flow_speed < 10) || (flow_speed > 1e5))
		{
			//cout << "filter the error is:" << feature_errors.at<float>(i, 0) << endl;
			continue;
		}
		//cout << "error is:" << feature_errors.at<float>(i, 0) << endl;
		//p0 ǰһ֡�������ǵ㣬������yolo�￴��û���غϵ�
		//p1 ��һ֡�������ǵ㣬������yolo�￴��û���غϵģ���Ӧ��cornersAͬһ��λ�õģ���ǰһ֡�ٳ�����
		//��ʶ������ͬһ�ֶ�����֤�����ٵ���  ����֡ͼ��Ķ�������һ֡��������  ��Щ�ǵ����Ϊý��

		Point p0 = Point(cvRound(cornersA.at<Vec2f>(i, 0)[0]), cvRound(cornersA.at<Vec2f>(i, 0)[1]));
		Point p1 = Point(cvRound(cornersB.at<Vec2f>(i, 0)[0]), cvRound(cornersB.at<Vec2f>(i, 0)[1]));

		//circle(drawimg2, p1, 2, Scalar(0, 0, 255), -1);  //scalar b g r
		/////circle(secondimg_orig, p1, 2, Scalar(0, 0, 255), -1);  //scalar b g r
		///line(imgC, p0, p1, Scalar(255, 0, 0),2);
		
		/////line(secondimg_orig, p0, p1, Scalar(0, 255, 255), 1);	

		////circle(show, p1, 2, Scalar(0, 0, 255), -1);  

		////line(show, p0, p1, Scalar(0, 255, 255), 1);

		circle(secondimg_orig, p1, 2, Scalar(0, 0, 255), -1);

		line(secondimg_orig, p0, p1, Scalar(0, 255, 255), 1);
	}
#ifdef SHOW
	//imshow("feature img", featureimg);
	//imshow("second img", secondimg);
	///imshow("drawimg2", drawimg2);
	///imshow("flow",imgC);   //flow��ͷ
	////waitKey(0);

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



//only for XueB
Mat  SamplesEnhance::roughLabelforXueB(Mat input, int left, int top, int right, int down, Rect &outRect)
{
	Mat show = input.clone();
	Rect rect(left, top, abs(left - right), abs(top - down));
	//rectangle(input, rect, Scalar(0, 0, 255), 2, LINE_8, 0);
	Rect rectobj = autoLableforXueB(input(rect));



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

		///////OutputLabelTXT(img, xmin, ymin, xmax, ymax, files_value[i], lable);
		//namedWindow(currentdir, 1);
		//resizeWindow(currentdir, Size(480, 320));
		//imshow(currentdir, result);
		imwrite(currentdir, result);
		//waitKey(0);
	}
}


void SamplesEnhance::roughLabelTestforXueB(string PICDIR, int init_left, int init_top, int init_right, int init_down)
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
		Mat result = roughLabelforXueB(img, init_left, init_top, init_right, init_down, outRect);

		int xmin = outRect.x;
		int ymin = outRect.y;
		int xmax = outRect.x + outRect.width;
		int ymax = outRect.y + outRect.height;
		int lable = 0;//nong fu shan quan
		OutputLabelTXT_keras(img, xmin, ymin, xmax, ymax, files_value[i], lable);

		imshow("show", result);

		//namedWindow(currentdir, 1);
		//resizeWindow(currentdir, Size(480, 320));
		//imshow(currentdir, result);
#ifdef SAVEPIC
		imwrite(currentdir, result);
#endif
		//waitKey(0);
	}
}

