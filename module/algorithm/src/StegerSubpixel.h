#ifndef STEGER_SUBPIXEL_H
#define STEGER_SUBPIXEL_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;
//***************************Zernike****************************************//
const Mat ZERPOLY00 = (Mat_<double>(5, 5) <<
	0.0219, 0.1231, 0.1573, 0.1231, 0.0219,
	0.1231, 0.1600, 0.1600, 0.1600, 0.1231,
	0.1573, 0.1600, 0.1600, 0.1600, 0.1573,
	0.1231, 0.1600, 0.1600, 0.1600, 0.1231,
	0.0219, 0.1231, 0.1573, 0.1231, 0.0219);

const Mat ZERPOLY11R = (Mat_<double>(5, 5) <<
	-0.0147, -0.0469, 0.0000, 0.0469, 0.0147,
	-0.0933, -0.0640, 0.0000, 0.0640, 0.0933,
	-0.1253, -0.0640, 0.0000, 0.0640, 0.1253,
	-0.0933, -0.0640, 0.0000, 0.0640, 0.0933,
	-0.0147, -0.0469, 0.0000, 0.0469, 0.0147);

const Mat ZERPOLY11I = (Mat_<double>(5, 5) <<
	0.0147, 0.0933, 0.1253, 0.0933, 0.0147,
	0.0469, 0.0640, 0.0640, 0.0640, 0.0469,
	0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
	-0.0469, -0.0640, -0.0640, -0.0640, -0.0469,
	-0.0147, -0.0933, -0.1253, -0.0933, -0.0147);

const Mat ZERPOLY20 = (Mat_<double>(5, 5) <<
	0.0177, 0.0595, 0.0507, 0.0595, 0.0177,
	0.0595, -0.0492, -0.1004, -0.0492, 0.0595,
	0.0507, -0.1004, -0.1516, -0.1004, 0.0507,
	0.0595, -0.0492, -0.1004, -0.0492, 0.0595,
	0.0177, 0.0595, 0.0507, 0.0595, 0.0177);

//***************************SM****************************************//
const Mat SM00 = (Mat_<double>(5, 5) <<
	0.0219, 0.1231, 0.1573, 0.1231, 0.0219,
	0.1231, 0.1600, 0.1600, 0.1600, 0.1231,
	0.1573, 0.1600, 0.1600, 0.1600, 0.1573,
	0.1231, 0.1600, 0.1600, 0.1600, 0.1231,
	0.0219, 0.1231, 0.1573, 0.1231, 0.0219);

const Mat SM10 = (Mat_<double>(5, 5) <<
	-0.0147, -0.0469, 0, 0.0469, 0.0147,
	-0.0933, -0.0640, 0, 0.0640, 0.0933,
	-0.1253, -0.0640, 0, 0.0640, 0.1253,
	-0.0933, -0.0640, 0, 0.0640, 0.0933,
	-0.0147, -0.0469, 0, 0.0469, 0.0147);

const Mat SM01 = (Mat_<double>(5, 5) <<
	0.0147, 0.0933, 0.1253, 0.0933, 0.0147,
	0.0469, 0.0640, 0.0640, 0.0640, 0.0469,
	0, 0, 0, 0, 0,
	-0.0469, -0.0640, -0.0640, -0.0640, -0.0469,
	-0.0147, -0.0933, -0.1253, -0.0933, -0.0147);

const Mat SM11 = (Mat_<double>(5, 5) <<
	-0.0098, -0.0352, 0, 0.0352, 0.0098,
	-0.0352, -0.0256, 0, 0.0256, 0.0352,
	0, 0, 0, 0, 0,
	0.0352, 0.0256, 0, -0.0256, -0.0352,
	0.0098, 0.0352, 0, -0.0352, -0.0098);

const Mat SM02 = (Mat_<double>(5, 5) <<
	0.0099, 0.0719, 0.1019, 0.0719, 0.0099,
	0.0194, 0.0277, 0.0277, 0.0277, 0.0194,
	0.0021, 0.0021, 0.0021, 0.0021, 0.0021,
	0.0194, 0.0277, 0.0277, 0.0277, 0.0194,
	0.0099, 0.0719, 0.1019, 0.0719, 0.0099);

const Mat SM20 = (Mat_<double>(5, 5) <<
	0.0099, 0.0194, 0.0021, 0.0194, 0.0099,
	0.0719, 0.0277, 0.0021, 0.0277, 0.0719,
	0.1019, 0.0277, 0.0021, 0.0277, 0.1019,
	0.0719, 0.0277, 0.0021, 0.0277, 0.0719,
	0.0099, 0.0194, 0.0021, 0.0194, 0.0099);
//***************************GM****************************************//
// 定义5x5邻域的r和c范围（文档中的 R = {-2,-1,0,1,2}, C = {-2,-1,0,1,2}）
const int R[5] = { -2, -1, 0, 1, 2 };
const int C[5] = { -2, -1, 0, 1, 2 };

// 定义10个正交多项式基函数指针类型
typedef double (*BasisFunc)(int, int);
// 正交多项式基函数声明
double P0(int r, int c);
double P1(int r, int c);
double P2(int r, int c);
double P3(int r, int c);
double P4(int r, int c);
double P5(int r, int c);
double P6(int r, int c);
double P7(int r, int c);
double P8(int r, int c);
double P9(int r, int c);


// 定义矩形信息结构体，存储矩形的长宽和中心点
struct RectangleInfo {
	float width;       // 矩形宽度
	float height;      // 矩形高度
	Point center;      // 中心点坐标
	Rect boundingRect; // 边界矩形
};
// 基函数数组（方便循环调用）
extern const BasisFunc basisFuncs[10];
// 预计算的基函数范数（sum(Pi^2)）
extern const vector<double> basisNorms;

// 基于TDDOP正交多项式拟合Facet模型系数
vector<double> fitFacetModel(const Mat& window);

// 从Facet模型系数计算Hessian矩阵（2x2）
Mat getHessianFromFacet(const vector<double>& coeffs);

// 提取初始边缘点（Canny边缘检测）
vector<Point> extractEdgePoints(const Mat& grayImage, double threshold1, double threshold2, int skip);

// 核心函数：结合Facet模型和Hessian矩阵提取亚像素边缘点
vector<Point2f> facetHessianSubpixel(const Mat& image, const vector<Point>& edgePoints);
int SubPixelByZernike1(const cv::Mat& src,
	const std::vector<cv::Point>& vecFinalKeypoints,
	std::vector<cv::Point2f>& group_pin_tough_corners);
int SubPixelByZernike(cv::Mat src, std::vector<cv::Point> vecFinalKeypoints, std::vector<cv::Point2f>& group_pin_tough_corners);
int CalulateCon(cv::Mat src, cv::Mat kernal, double& value);
int SubPixelBySM(cv::Mat src, std::vector<cv::Point> vecFinalKeypoints, std::vector<cv::Point2f>& group_pin_tough_corners);
int SubPixelByOurMethod(cv::Mat src, std::vector<std::vector<cv::Point2i>> vvPoints, std::vector<cv::Point2f>& vvSubPixelPoints);

//bool detectRectangles(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea);
#endif // STEGER_SUBPIXEL_H