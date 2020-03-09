#include "stdafx.h"
//#include "SamplesEnhance.h"
#include <sys/types.h> 
#include <sys/stat.h> 
#include<io.h>
#include<time.h>
#include <iostream> 
#include <string.h> 
#include <stdio.h> 
#include <fcntl.h> 
#include <stdlib.h>  
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include<fstream>
using namespace std;
using namespace cv;

//using namespace samle;


void getFiles1(string path, vector<string>& files)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles1(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(path + "\\" + fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
void getFiles2(string path, vector<string>& files, vector<string> &ownname)
{
	/*files存储文件的路径及名称(eg.   C:\Users\WUQP\Desktop\test_devided\data1.txt)
	ownname只存储文件的名称(eg.     data1.txt)*/
	long   hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles2(p.assign(path).append("\\").append(fileinfo.name), files, ownname);
			}
			else
			{
				files.push_back(path + "\\" + fileinfo.name);
				ownname.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

string pathConvert_Single2Double(string& s) {
	string news;
	string::size_type pos = 0;
	while ((pos = s.find('\\', pos)) != string::npos) {
		s.insert(pos, "\\");
		pos = pos + 2;
	}
	news = string(s);
	return  news;
}

vector<string> listFileFromDIR(string  filePath)
{
	//string  filePath = "E:\\Xscx2019\\test";
	vector<string> files;
	vector<string> files_value;

	getFiles1(filePath, files);
	int files_num = files.size();
	for (int i = 0; i < files_num; i++)
	{
		files_value.push_back(pathConvert_Single2Double(files[i]));
	}
	return files_value;
}



string changeJPGtoTXT(string jpg)
{
	string dir_txt = string(jpg.c_str());
	//dir_txt = jpg.c_str();

	string::size_type point = dir_txt.rfind("jpg");

	dir_txt.replace(point, 3, "txt");
	return dir_txt;
}


string changeJPGtoTXT2(string jpg)
{
	string dir_txt = string(jpg.c_str());
	//dir_txt = jpg.c_str();

	string::size_type point = dir_txt.rfind("jpg");

	dir_txt.replace(point, 3, "txt");
	return dir_txt;
}

string renameJPG(string jpg)
{
	string dir_txt = string(jpg.c_str());
	//dir_txt = jpg.c_str();
	//fix JPG jpg
	//string::size_type point0 = dir_txt0.rfind("JPG");
	//dir_txt0.replace(point0, 3, "jpg");   

	//string dir_txt = string(dir_txt0.c_str());

	string::size_type point = dir_txt.rfind("jpg");

	dir_txt.replace(point-1, 5, "g.jpg");
	return dir_txt;
}


string getDir(string jpg)
{
	string dir_txt = string(jpg.c_str());
	//string news;
	string::size_type pos = 0;
	while ((pos = dir_txt.find('\\\\', pos)) != string::npos) {
		//s.insert(pos, "\\");
		
		
		pos += 2;
		if ((dir_txt.find('\\\\', pos)) == string::npos)
			break;
		
	}
	cout << pos << endl;
	int residue = dir_txt.length() - pos;
	dir_txt.erase(pos, residue);
	//news = string(s);
	//return  news;

	//string dir_txt = string(jpg.c_str());
	//string::size_type point = dir_txt.rfind(".jpg");
	//dir_txt.replace(point, 4, "");
	/////dir_txt.replace(pos, 1, "K");
	cout << dir_txt << endl;
	return dir_txt;
}
void OutputLabelTXT(Mat imgO,double xmin, double ymin, \
	double xmax, double ymax,string pic_dir,int lable)
{
	ofstream outfile;
	string dir_txt = changeJPGtoTXT(pic_dir);
	outfile.open(dir_txt.c_str(), ios::out);
	outfile << imgO.size().width << ' ';
	outfile << imgO.size().height << ' ';
	outfile << xmin << ' ';
	outfile << xmax << ' ';
	outfile << ymin << ' ';
	outfile << ymax << ' ';
	//outfile<<argv[2]<<' ';
	outfile << lable << ' ';
	outfile << endl;
	outfile.close();
}

void OutputLabelTXT_keras(Mat imgO, double xmin, double ymin, \
	double xmax, double ymax, string pic_dir, int lable)
{
	ofstream outfile;
	string dir_txt = changeJPGtoTXT(pic_dir);
	outfile.open(dir_txt.c_str(), ios::out);
	//outfile << imgO.size().width << ' ';
	//outfile << imgO.size().height << ' ';
	outfile << xmin << ',';
	outfile << ymin << ',';
	outfile << xmax << ',';
	outfile << ymax << ',';
	//outfile<<argv[2]<<' ';
	outfile << lable;
	outfile << endl;
	outfile.close();
}