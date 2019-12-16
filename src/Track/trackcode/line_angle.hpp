#ifndef LINE
#define LINE
#include "common.h"
#include "equation_set.hpp"
//直线 向量
typedef struct coordinate
{
	double x;
	double y;
}array_coordinate;


array_coordinate vector_coordinate(Vec4d line)
{
	array_coordinate arr_coor;
	arr_coor.x = line[2] - line[0];
	arr_coor.y = line[3] - line[1];
	return arr_coor;
}
class line_angle 
{
private:
	float x;
	float y;
	array_coordinate cor;
	array_coordinate vector_coordinate(Vec4f line)
	{
		array_coordinate arr_coor;
		arr_coor.x = line[2] - line[0];
		arr_coor.y = line[3] - line[1];
		cor.x = arr_coor.x;
		cor.y = arr_coor.y;
		return arr_coor;
	};
public:
	line_angle(Vec4f line)
	{
		vector_coordinate(line);
	};
	float vector_slope()
	{
		return cor.y / cor.x;
	}
 };



double inner_product(Vec4d line1, Vec4d line2)
{
	array_coordinate aline = vector_coordinate(line1);
	array_coordinate bline = vector_coordinate(line2);
	//cout << "aline.x" << aline.x << endl;
	//cout << "aline.y" << aline.y << endl;

	//cout << "bline.x" << bline.x << endl;
	//cout <<" bline.y" << bline.y << endl;
	return aline.x * bline.x + aline.y * bline.y;
}

double norm(Vec4d line0)
{
	array_coordinate line = vector_coordinate(line0);
	return sqrt(line.x * line.x + line.y *line.y);
}


double cos_value_to_horizon(Vec4d line0)
{
	array_coordinate line = vector_coordinate(line0);
	return line.x / norm(line0);
}

double sin_value_to_horizon(Vec4d line0)
{
	array_coordinate line = vector_coordinate(line0);
	return line.y / norm(line0);
}

//r = x*cos(theta) + y*sin(theta);
//sin（π / 2－α） = cosα
//cos（π / 2－α） = sinα
double normal_equation_para(Vec4d line0)
{
	double x = line0[0], y = line0[1];
	//cout << "x" << x << endl;
	//cout << "y" << y << endl;
	double cosa = abs(sin_value_to_horizon(line0)), sina = abs(cos_value_to_horizon(line0));  //加ABS修正 解决上挑斜线sin值是负数的问题
																							 //cout << "cosa" << cosa << endl;
																							 //cout << "sina" << sina << endl;
	cout << endl;
	//cout << "sinnnnnnnnnn" << sin_value_to_horizon(line0) << endl;
	//cout <<"cossssssssss"<< cos_value_to_horizon(line0) << endl;
	if (sin_value_to_horizon(line0) < 0)
	{
		//cout << "polar angle:" << asin(sina) * 180 / 3.1415926 << endl;
		double r = x*cosa + y*sina;
		return r;
	}
	else
	{
		double a = asin(sina) * 180 / 3.1415926;
		a = 180 - a;
		//cout << "polar angle:" << a << endl;
		a = a*3.1415926 / 180.0;

		double r = x*cos(a) + y*sin(a);
		return r;
	}

	//float a = 105*3.1415926/180;
	//float r = x*cos(a) + y*sin(a);
	//float r = abs(-x*sina + y*cosa);
	//float r = x*cosa + y*sina;

}


//用这个来判断两个直线是否平行 顺势用距离判断是否足够小
double intersection_angle(Vec4d aline, Vec4d bline)
{/*
	cout << endl;
	cout << "a[0]" << aline[0] << endl;
	cout << " a[1]" << aline[1] << endl;
	cout << "a[2]" << aline[2] << endl;
	cout << "a[3]" << aline[3] << endl;

	cout << "a[0]" << bline[0] << endl;
	cout << "a[1]" << bline[1] << endl;
	cout << "a[2]" << bline[2] << endl;
	cout << "a[3]" << bline[3] << endl;

	cout << "inner_product" << inner_product(aline, bline) << endl;
	cout << "norm" << norm(aline) << endl;
	cout << "norm" << norm(bline) << endl;
	cout << "test" << inner_product(aline, bline) / (norm(aline)*norm(bline)) << endl;
	*/
	double deg = acos(inner_product(aline, bline) / (norm(aline)*norm(bline)));
	deg = deg / 3.1415926 * 180;
	return deg;
}
//r=x*cos(theta)+y*sin(theta);
//两点式方程 (y-y1)/(y2-y1)=(x-x1)/(x2-x1);

//求两线交点  (y-y1)/(y2-y1)=(x-x1)/(x2-x1);和x=(x1+x2)/2
vector<Point> intersectionPoint_verticle(Vec4d line1, Vec4d line2)
{
	float xmid, yline1, yline2;
	xmid = (line1[0] + line2[0]) / 2;
	yline1 = (xmid - line1[0]) / vector_coordinate(line1).x*vector_coordinate(line1).y + line1[1];
	yline2 = (xmid - line2[0]) / vector_coordinate(line2).x*vector_coordinate(line2).y + line2[1];
	vector<Point> res(2);
	res[0] = Point(xmid, yline1);
	res[1] = Point(xmid, yline2);

	return res;

}
//r=x*cos(theta)+y*sin(theta);
//两点式方程 (y-y1)/(y2-y1)=(x-x1)/(x2-x1);

//求两线交点  (y-y1)/(y2-y1)=(x-x1)/(x2-x1);和x=(x1+x2)/2

//y=kx+b
//b=y1-x1*k
void resolveLineKXEqution(Vec4d line, double &k, double &b)
{
	double x1 = line[0];
	double y1 = line[1];
	double x2 = line[2];
	double y2 = line[3];
	k = (y2 - y1) / (x2 - x1);
	b = y1 - k*x1;
}

double solutionYlineKXEqution(double k, double b, double x)
{
	double y= k*x + b;
	return y;
}
double solutionXlineKXEqution(double k, double b, double y)
{
	double x = (y-b)/k;
	return x;
}



Vec3f line_para_extract(Vec4f line1)
{
	//(x - line1[0]) / (y - line1[1]) = vector_coordinate(line1).x / vector_coordinate(line1).y
	//x- (vector_coordinate(line1).x / vector_coordinate(line1).y)*y
	//	= line1[0]-(vector_coordinate(line1).x / vector_coordinate(line1).y)*line1[1]

	//x- (vector_coordinate(line1).x / vector_coordinate(line1).y)*y
	//	= line1[0]-(vector_coordinate(line1).x / vector_coordinate(line1).y)*line1[1]

	//(vector_coordinate(line1).y / vector_coordinate(line1).x)*x-y
	//	= (vector_coordinate(line1).y / vector_coordinate(line1).x)*line1[0]- line1[1]

	float x, y;

	if (abs(vector_coordinate(line1).x) < 15)   //垂线  y=a
	{
		Vec3f para;
		para[0] = 1;
		para[1] = 0;
		para[2] = line1[0];
		return para;
	}
	else if (abs(vector_coordinate(line1).y) < 15)    //平线  x=a
	{
		Vec3f para;
		para[0] = 0;
		para[1] = -1;
		para[2] = -line1[1];
		return para;
	}
	else    //平线方法
	{
		////////cout << "xielvdaoshu" << vector_coordinate(line1).x / vector_coordinate(line1).y << endl;
		Vec3f para;
		para[0] = (vector_coordinate(line1).y / vector_coordinate(line1).x);
		para[1] = -1;
		para[2] = (vector_coordinate(line1).y / vector_coordinate(line1).x)*line1[0] - line1[1];
		return para;
	}
	/*
	else      //垂线方法
	{
	cout << "xielvdaoshu" << vector_coordinate(line1).x / vector_coordinate(line1).y << endl;
	Vec3f para;
	para[0] = 1;
	para[1] = -(vector_coordinate(line1).x / vector_coordinate(line1).y);
	para[2] = line1[0] - (vector_coordinate(line1).x / vector_coordinate(line1).y)*line1[1];
	return para;
	}
	*/
}
//extern vector<float> equation_set_solve(float martix[][3], int n, int m);
//vector<Point>
//resolve the intersection
Point intersectionPoint_true(Vec4f line1, Vec4f line2)
{
	//float xmid, yline1, yline2;
	Vec3f line1p = line_para_extract(line1);
	Vec3f line2p = line_para_extract(line2);
	float matrix[2][3];
	matrix[0][0] = line1p[0]; matrix[0][1] = line1p[1]; matrix[0][2] = line1p[2];
	matrix[1][0] = line2p[0]; matrix[1][1] = line2p[1]; matrix[1][2] = line2p[2];
	//matrix[1][0] = 1; matrix[1][1] = 0.1; matrix[1][2] = 0.2;
///////	cout << "解之前" << endl;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 3; j++)
			printf("%-10.2f", matrix[i][j]);
		printf("\n");
	}
	//////cout << "解之后" << endl;
	vector<float> res(2);
	res = equation_set_solve(matrix, 2, 3);
	///////cout << "x=" << res[0] << endl;
	///////cout << "y=" << res[1] << endl;
	Point result(res[0], res[1]);
	return result;
}


//typedef vector<float,4> Vec4f 
double distance_between_lines(Vec4d line1, Vec4d line2)
{
	//倾斜度可以弄到界面上当参数
	//if ((intersection_angle(line1, line2) > 3) && (intersection_angle(line1, line2) < 177))
	if ((intersection_angle(line1, line2) > 15) && (intersection_angle(line1, line2) < 160))
	{
		cout << "not paralell,return " << endl;
		return -1;
	}
	double distance = abs(normal_equation_para(line1) - normal_equation_para(line2));
	return distance;  //缺一个返回值导致debug和release不一致
	/*
	float distance;
	if ((acos(cos_value_to_horizon(line1)) < 3) || (acos(cos_value_to_horizon(line1)) > 177))
	{
	distance = abs(line1[1] - line2[1]);
	}
	else   //运用了两点式直线方程
	{
	float xmid, yline1, yline2;
	xmid = (line1[0] + line2[0]) / 2;
	yline1 = (xmid - line1[0]) / vector_coordinate(line1).x*vector_coordinate(line1).y + line1[1];
	yline2 = (xmid - line2[0]) / vector_coordinate(line2).x*vector_coordinate(line2).y + line2[1];
	distance = abs(yline1 - yline2)*cos_value_to_horizon(line1);
	}
	return distance;
	*/
}


float distance_between_lines_aver(Vec4f bukle_line, Vec4f boxedge_line)
{
  float distance;
 // bukle_line[0]  bukle_line[1]   bukle_line[2]   bukle_line[3]
  float bukle_line_y;
  float boxedge_line_y;
  float sum = 0;
  float num = bukle_line[2] - bukle_line[0] + 1;
  for (int x = bukle_line[0]; x < bukle_line[2]; x++)
  {
    bukle_line_y = (x - bukle_line[0]) / vector_coordinate(bukle_line).x*vector_coordinate(bukle_line).y + bukle_line[1];
	boxedge_line_y = (x - boxedge_line[0]) / vector_coordinate(boxedge_line).x*vector_coordinate(boxedge_line).y + boxedge_line[1];
	float delta = bukle_line_y - boxedge_line_y;
	sum = sum + delta;
  }
  float aver = sum / num;
  /*
  else   //运用了两点式直线方程
  {
  float xmid, yline1, yline2;
  xmid = (line1[0] + line2[0]) / 2;
  yline1 = (xmid - line1[0]) / vector_coordinate(line1).x*vector_coordinate(line1).y + line1[1];
  yline2 = (xmid - line2[0]) / vector_coordinate(line2).x*vector_coordinate(line2).y + line2[1];
  distance = abs(yline1 - yline2)*cos_value_to_horizon(line1);
  }
  */
  return aver;
  
}

vector<Vec4d> convertRoiCoordinate(vector<Vec4d> Lines, Rect Roi)
{
	vector<Vec4d> Lines_convert(Lines.size());
	for (size_t i = 0; i < Lines.size(); i++)
	{

		Lines_convert[i][0] = Lines[i][0] + Roi.x;
		Lines_convert[i][1] = Lines[i][1] + Roi.y;

		Lines_convert[i][2] = Lines[i][2] + Roi.x;
		Lines_convert[i][3] = Lines[i][3] + Roi.y;
	}
	return Lines_convert;

}

#endif