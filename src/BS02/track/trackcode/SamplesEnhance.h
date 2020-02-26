#pragma once
#ifndef SMPLE_ENHANCE
#define SMPLE_ENHANCE
#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/dnn.hpp>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <opencv2/core/mat.hpp>

#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include<fstream>
#include <cstdlib>

#include <fcntl.h>
#include<io.h>
#include<time.h>
#include <vector>

#include "Camera.h"
using namespace cv;
using namespace std;
using namespace dnn;
//#define SHOW
#define SAVEPIC


//namespace sample {
class SamplesEnhance
{
private:
	//lable
	const int lable = 2;
	//pic size
	const int resize_width = 600, resize_height = 500;
	bool bgNeedUpdated = true;
	//bool bgNeedUpdated = false;
	const int BG_STUDY_NUM = 60;
	double high_threadhold = 6.0;
	double low_threadhold = 6.0;
	Rect rectFlag;
	vector<Mat> bgVector;
	//string BG_HIGH_THRESHOLD_DIR = "E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\highThreadhold.tiff";
	//string BG_LOW_THRESHOLD_DIR = "E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\lowThreadhold.tiff";

	string BG_HIGH_THRESHOLD_DIR = "E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\highThreadhold.jpg";
	string BG_LOW_THRESHOLD_DIR = "E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\lowThreadhold.jpg";
	//useful element for morph
	Mat element_5 = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat element_3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element_7 = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat element_13 = getStructuringElement(MORPH_RECT, Size(13, 13));
	Mat element_25 = getStructuringElement(MORPH_RECT, Size(25, 25));
	Mat element_37 = getStructuringElement(MORPH_RECT, Size(37, 37));
	Mat element_19 = getStructuringElement(MORPH_RECT, Size(19, 19));

	Mat element_ecllipse_5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	Mat element_ecllipse_3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	Mat element_ecllipse_7 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	Mat element_ecllipse_13 = getStructuringElement(MORPH_ELLIPSE, Size(13, 13));
	Mat element_ecllipse_19 = getStructuringElement(MORPH_ELLIPSE, Size(19, 19));
	Mat element_ecllipse_25 = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
	Mat element_ecllipse_37 = getStructuringElement(MORPH_ELLIPSE, Size(37, 37));
	//HSV color space
	//green
	int greenLowH = 35;
	int greenHighH = 77;
	int greenLowS = 43;
	int greenHighS = 255;
	int greenLowV = 46;
	int greenHighV = 255;
	//yellow
	int yellowLowH = 26;
	int yellowHighH = 34;
	int yellowLowS = 43;
	int yellowHighS = 255;
	int yellowLowV = 46;
	int yellowHighV = 255;
	//gray

	int grayLowH = 0;
	int grayHighH = 180;
	int grayLowS = 0;
	int grayHighS = 43;
	int grayLowV = 46;
	int grayHighV = 220;

	//blue
	int blueLowH = 100;
	int blueHighH = 124;
	int blueLowS = 43;
	int blueHighS = 255;
	int blueLowV = 46;
	int blueHighV = 255;

	

	int redLowH = 0;
	int redHighH = 12;
	int redLowS = 43;
	int redHighS = 255;
	int redLowV = 46;
	int redHighV = 255;



	const int nongfu_width = 180;
	const int nongfu_height = 38;

	const int xuebi_width = 210;
	const int xuebi_height =60;
	

	Point GetRectCenterPoint(Rect rect);

	Mat gamaPROC(Mat img);

	

	
	void frameSub(Mat img_pre, Mat img_later);
	
	void salt(Mat& image, int n);
	
	Mat mysobel(Mat img_pre);
	Mat sharp_laplace(Mat input);

	/////avg bacmodel
	/*******************************/
	Mat IavgF, IdiffF;  // average frame  differ frame
	Mat	IprevF; //pre frame
	Mat IhiF, IlowF;   //high threadhold  low threadhold
	Mat Iscratch, Iscratch2;   //temp

	Mat Igray1, Igray2, Igray3;   //divide to 3 channels
	Mat Ilow1, Ilow2, Ilow3;
	Mat Ihi1, Ihi2, Ihi3;
	Mat Imaskt;

	float Icount;
	void AllocateImages(Mat I);
	void avgBackground(Mat I);

	void createModelsfromStats();
	void setHighThreshold(float scale);
	void setLowThreshold(float scale);
	void backgroundDiff(Mat I, Mat &Imask,vector<Rect> &rectarray);
	void learnBackground(string PICDIR);
public:
	void gamaTest(string PICDIR);
	Mat roughLabel(Mat input,int left,int top,int right,int down, Rect &outRect);
	void roughLabelTest(string PICDIR, int init_left, int init_top, int init_right, int init_down);
	void autoLableforColorTest(string PICDIR);
	Rect autoLableforColor(Mat &imgO);
	void frameSubTest(string PICDIR);
	void saltTest(string PICDIR, int noise_point_num);
	void resizePicTest(string PICDIR, int resize_width, int resize_height);
	void foreGroundSegmentTest(string PICDIR);
	void LKlightflow_trackTest(string PICDIR);
	void LKlightflow_track(Mat featureimg, Mat &secondimg_orig);
	void LKlightflow_trackCamTest();
	Rect autoLableforXueB(Mat &imgO);
	vector<String> getOutputsNames(Net&net);
	void  drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
	void  postprocess(Mat& frame, const vector<Mat>& outs, float confThreshold, float nmsThreshold);
	int  dnnTest();
	void roughLabelTestforXueB(string PICDIR, int init_left, int init_top, int init_right, int init_down);
	Mat  roughLabelforXueB(Mat input, int left, int top, int right, int down, Rect &outRect);
	void studyBackgroundFromCam(camera cam);
	void loadBackgroundModel();
	void  getPicFromCam(camera cam);
	void SamplesEnhance::findrColor(Mat &imgO, Rect &rectout);
	void SamplesEnhance::learnBackGroundFromVec(vector<Mat> bgVector);
};

//}



//file methods 
extern void getFiles2(string path, vector<string>& files, vector<string> &ownname);
extern string pathConvert_Single2Double(string& s);
extern void getFiles1(string path, vector<string>& files);
extern vector<string> listFileFromDIR(string  filePath);
extern string changeJPGtoTXT(string jpg);
extern string renameJPG(string jpg);
extern  string getDir(string jpg);
extern void OutputLabelTXT(Mat imgO, double xmin, double ymin, \
	double xmax, double ymax, string pic_dir, int lable);
extern void OutputLabelTXT_keras(Mat imgO, double xmin, double ymin, \
	double xmax, double ymax, string pic_dir, int lable);
extern string changeJPGtoTXT2(string jpg);

#endif
