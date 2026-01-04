#ifndef MARK_TEACH_H
#define MARK_TEACH_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// 定义矩形信息结构体，存储矩形的长宽和中心点
struct RectangleInfo {
    float width;
    float height;
	float width_inner;       // 小矩形
	float height_inner;      //
    float width_outer;       // 大矩形
    float height_outer;      // 
	Point center;      // 中心点坐标
	Rect boundingRect; // 边界矩形
    RotatedRect rotatedRect;//旋转矩形
    float pitch_inner;
    float pitch_outer;
};
struct CrossMarkInfo {
    Point2f center;          // 十字中心坐标
    double horizontalArmLen; // 水平臂总长度
    double verticalArmLen;   // 垂直臂总长度
    double horizontalArmWidth; // 水平臂宽度
    double verticalArmWidth;   // 垂直臂宽度
    Rect boundingRect;       // 包围矩形
};
bool detectAIM(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea);
bool detectFIF(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea);
bool detectBoxinBox(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea);
bool detectBarinBar(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea);
bool detectWaferCrossMarks(const Mat& image, vector<CrossMarkInfo>& crossInfos, int& minFeatureSize);

Point2f getLineIntersection(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4);
void getLengthAndWidth(const RotatedRect& rect, float& height, float& width, float& angle);
vector<vector<Point2f>> groupRectanglesByAngle(const vector<RotatedRect>& rects, Point2f& totalrect);
vector<float> calculateGroupDistances(const vector<Point2f>& centers, bool sortByX);
float calculateDistance(const Point2f& p1, const Point2f& p2);
#endif // STEGER_SUBPIXEL_H