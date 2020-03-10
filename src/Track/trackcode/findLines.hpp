#include "common.h"
//Ѱ��ͼ��������ĳ�������һ����
const int N = 8;
bool findNextPoint(vector<Point> &_neighbor_points, Mat &_image, Point _inpoint, int flag, Point& _outpoint, int &_outflag)
{
	int i = flag;
	int count = 1;
	bool success = false;

	while (count <= (N-1))
	//while (count <= 4)
	{
		Point tmppoint = _inpoint + _neighbor_points[i];
		if (tmppoint.x > 0 && tmppoint.y > 0 && tmppoint.x < _image.cols&&tmppoint.y < _image.rows)
		{
			if (_image.at<uchar>(tmppoint) == 255)
			{
				_outpoint = tmppoint;
				_outflag = i;
				success = true;
				_image.at<uchar>(tmppoint) = 0;  //��ֹ�ظ��ҵ� ����ԭͼ
				break;
			}
		}
		if (count % 2)
		{
			i += count;
			if (i > (N - 1))
			//if (i > 4)
			{
				i -= N;
				//i -= 5;
			}
		}
		else
		{
			i += -count;
			if (i < 0)
			{
				i += N;
				//i += 5;
			}
		}
		count++;
	}
	return success;
}
//Ѱ��ͼ���ϵĵ�һ����
bool findFirstPoint(Mat &_inputimg, Point &_outputpoint)
{
	bool success = false;
	for (int i = 0; i < _inputimg.rows; i++)
	{
		uchar* data = _inputimg.ptr<uchar>(i);
		for (int j = 0; j < _inputimg.cols; j++)
		{
			if (data[j] == 255)
			{
				success = true;
				_outputpoint.x = j;
				_outputpoint.y = i;
				data[j] = 0;
				break;
			}
		}
		if (success)
			break;
	}
	return success;
}


//����  �ҵ�A B�㸽���� ͼ���ϵĵ�
bool findFirstPoint(Mat &_inputimg, Point &_inputpoint, Point &_outputpoint)
{
	bool success = false;
	const int scope = 20;
	for (int i = _inputpoint.y -scope; (i < _inputpoint.y+ scope)&&(i>1); i++)//i<1����
	{
		uchar* data = _inputimg.ptr<uchar>(i);
		for (int j = _inputpoint.x - scope; j < _inputpoint.x + scope; j++)
		{
			if (data[j] == 255)
			{
				success = true;
				_outputpoint.x = j;
				_outputpoint.y = i;
				data[j] = 0;
				break;
			}
		}
		if (success)
			break;
	}
	return success;
}



//Ѱ������ 
void findLines(Mat &_inputimg, vector<deque<Point>> &_outputlines)
{
	vector<Point> neighbor_points = { Point(-1,-1),Point(0,-1),Point(1,-1),Point(1,0),Point(1,1),Point(0,1),Point(-1,1),Point(-1,0) };
	
	Point first_point;
	while (findFirstPoint(_inputimg, first_point))
	{
		deque<Point> line;
		line.push_back(first_point);
		//���ڵ�һ���㲻һ�����߶ε���ʼλ�ã�˫����
		Point this_point = first_point;
		int this_flag = 0;
		Point next_point;
		int next_flag;
		while (findNextPoint(neighbor_points, _inputimg, this_point, this_flag, next_point, next_flag))
		{
			line.push_back(next_point);
			this_point = next_point;
			this_flag = next_flag;
		}
		//����һ��
		this_point = first_point;
		this_flag = 0;
		//cout << "flag:" << this_flag << endl;
		while (findNextPoint(neighbor_points, _inputimg, this_point, this_flag, next_point, next_flag))
		{
			line.push_front(next_point);
			this_point = next_point;
			this_flag = next_flag;
		}
		if (line.size() > 10)
		{
			_outputlines.push_back(line);
		}
	}
}

//Ѱ������ 
void findLines(Mat &_inputimg, vector<deque<Point>> &_outputlines, Point first_point,bool find_direction=true)
{
	vector<Point> neighbor_points(N);
	
	if(find_direction==true)
	{//find left     //��ô�ı䶼û�� ˫����ı��Ǳ����  //��122�ϱ��ֲ���
		vector<Point> tmp = { Point(-1,-1),Point(0,-1),Point(1,-1),Point(1,0),\
			Point(1,1),Point(0,1),Point(-1,1),Point(-1,0) };
		//vector<Point> tmp = { Point(0,-1),Point(-1,-1),Point(-1,0),Point(-1,1),\
			Point(0,1), Point(1,1),Point(1,0),Point(1,-1),Point(0, -1), Point(1, -1), Point(1, 0), \
			Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0), Point(-1, -1),Point(-1, 2), Point(0, 2), Point(1, 2), Point(-1, -2),Point(0, -2), Point(1, -2) };
		neighbor_points.assign(tmp.begin(), tmp.end());
	}
	else
	{//find right   //��121�ϱ��ֲ���
		vector<Point> tmp = { Point(0,-1),Point(1,-1),Point(1,0),\
			Point(1,1),Point(0,1),Point(-1,1),Point(-1,0) ,Point(-1, -1),
			Point(-1,2), Point(0, 2), Point(1, 2), Point(-1, -2),
			Point(0, -2), Point(1, -2)};
		//vector<Point> tmp = { Point(-1,-1),Point(0,-1),Point(1,-1),Point(1,0),\
			Point(1,1),Point(0,1),Point(-1,1),Point(-1,0) };
		neighbor_points.assign(tmp.begin(), tmp.end());
	}
	
	/* ����
	if (find_direction == true)
	{//find left
		vector<Point> tmp = { Point(-1,-1),Point(0,-1),\
			Point(0,1),Point(-1,1),Point(-1,0) };
		neighbor_points.assign(tmp.begin(), tmp.end());
	}
	else
	{//find right
		vector<Point> tmp = { Point(0,-1),Point(1,-1),Point(1,0),\
			Point(1,1),Point(0,1) };
		neighbor_points.assign(tmp.begin(), tmp.end());
	}
	*/

	deque<Point> line;
	line.push_back(first_point);
	//���ڵ�һ���㲻һ�����߶ε���ʼλ�ã�˫����
	Point this_point = first_point;
	int this_flag = 0;
	Point next_point;
	int next_flag;
	while (findNextPoint(neighbor_points, _inputimg, this_point, this_flag, next_point, next_flag))
	{
		line.push_back(next_point);
		this_point = next_point;
		this_flag = next_flag;
	}
	
	//����һ��
	this_point = first_point;
	this_flag = 0;
	//cout << "flag:" << this_flag << endl;
	while (findNextPoint(neighbor_points, _inputimg, this_point, this_flag, next_point, next_flag))
	{
		line.push_front(next_point);
		this_point = next_point;
		this_flag = next_flag;
	}
	
	if (line.size() > 10)
	{
		_outputlines.push_back(line);
	}
	
}

//���ȡɫ ���ڻ��ߵ�ʱ��
Scalar random_color(RNG& _rng)
{
	int icolor = (unsigned)_rng;
	return Scalar(icolor & 0xFF, (icolor >> 8) & 0xFF, (icolor >> 16) & 0xFF);
}





/*
int main()
{
	Mat image = imread("images\\2.bmp");
	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);
	vector<deque<Point>> lines;
	findLines(gray, lines);
	cout << lines.size() << endl;
	//draw lines
	Mat draw_img = image.clone();
	RNG rng(123);
	Scalar color;
	for (int i = 0; i < lines.size(); i++)
	{
		color = random_color(rng);
		for (int j = 0; j < lines[i].size(); j++)
		{
			draw_img.at<Vec3b>(lines[i][j]) = Vec3b(color[0], color[1], color[2]);
		}
	}
	imshow("draw_img", draw_img);
	imwrite("images\\draw_img.bmp", draw_img);
	waitKey(0);
	system("pause");
	return 0;
}
*/