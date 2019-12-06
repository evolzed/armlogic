#pragma once
#include "MvCameraControl.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
//#include <Windows.h>
#include <process.h>
#include <conio.h>
#include "string.h"
#include <iostream>
using namespace cv;
using namespace std;

class camera
{

	int nRet = MV_OK;
	void* handle = NULL;
	unsigned char * pData;
	MV_FRAME_OUT_INFO_EX stImageInfo = { 0 };
public:
	void init();
	Mat getImage();
	void close();

};
