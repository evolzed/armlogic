// myImgprocTools.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "SamplesEnhance.h"

/*not  adapt to TransParent bottle auto label,gray background*/


//#include "methodlabrary.h"

string  PICDIR_autolable = "E:\\Xscx2019\\test";
string  PICDIR_resize = "E:\\Xscx2019\\OPENCV_PROJ\\picForMotion";
string  PICDIR_gama = "E:\\DataShare\\已经标注图片\\yuan\\20191119\\2019-11-19_data - 副本";
//string  PICDIR_motion = "E:\\Xscx2019\\OPENCV_PROJ\\picForMotion\\orig\\1";
string  PICDIR_motion = "E:\\Xscx2019\\OPENCV_PROJ\\picForMotion\\2\\2";
//E:\Xscx2019\OPENCV_PROJ\picForMotion\2\2
string  PICDIR_noise = "E:\\DataShare\\已经标注图片\\yuan\\20191120noise\\2019-11-20";
string PICDIR_ground = "E:\\Xscx2019\\vedio\\5";
string  PICDIR_roughlabel = "E:\\DataShare\\DataSet6";

int main(int argc, char **argv)
{
	SamplesEnhance obj;
	//obj.autoLableTest(PICDIR_autolable);
	//obj.resizePicTest(PICDIR_resize,640,480); //有返回值一定要写返回值
	//obj.gamaTest(PICDIR_gama); //有返回值一定要写返回值
	//obj.frameSubTest(PICDIR_motion);

	//obj.saltTest(PICDIR_noise, 1000);
	//obj.foreGroundSegmentTest(PICDIR_ground);
	//obj.LKlightflow_trackTest(PICDIR_ground);
	//obj.LKlightflow_trackCamTest();

	//void SamplesEnhance::roughLabelTest(string PICDIR, int init_left, int init_top, int init_right, int init_down)
	obj.roughLabelTest(PICDIR_roughlabel, 437, 411, 656, 542);
	//waitKey(0);
	//char c;
	//cin >> c;
	return 0;

}

