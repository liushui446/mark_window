#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "MarkTeach.h"
using namespace cv;
using namespace std;
const double PI = 3.14159265358979323846;


bool detectAIM(const Mat& image, vector<RectangleInfo>& rectInfos, int&minArea ) {
    // 3. 二值化处理
    Mat binary;
    threshold(image, binary, 127, 255, THRESH_BINARY_INV | THRESH_OTSU); // 使用OTSU自动阈值
    //// 4. 形态学操作优化轮廓
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel); // 填充小空洞

    cv::Mat medianFilteredImage;
    cv::medianBlur(image, medianFilteredImage, 3);
    cv::Mat preprocessedImage;
    cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

    // 使用双边滤波减少噪声
    cv::Mat smoothedImage2, edges2;
    cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
    cv::Canny(binary, edges2, 170, 200, 3, true);
    // 5. 轮廓检测：使用RETR_CCOMP获取所有层次轮廓（包括内部轮廓）
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // RETR_CCOMP模式可以获取所有外层和内层轮廓
    //findContours(edges2, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    findContours(edges2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //findContours(edges2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {

        return false;
    }
    // 使用所有轮廓点计算外接矩形
    vector<Point> allPoints;
    for (const auto& contour : contours) {
        allPoints.insert(allPoints.end(), contour.begin(), contour.end());
    }

    // 使用boundingRect计算外接矩形
    RotatedRect totalrect = minAreaRect(allPoints);
    // 6. 处理轮廓，筛选内部轮廓（重点）
    rectInfos.clear();
    //方法2旋转矩形
    // 定义一个向量用于存储检测到的旋转矩形
    std::vector<RotatedRect> detectedRects;
    double maxPerimeter = 0.0;
    for (size_t i = 0; i < contours.size(); i++) {
        double perimeter = cv::arcLength(contours[i], true);
        if (perimeter > maxPerimeter) {
            maxPerimeter = perimeter;
        }
    }

    // 计算最小周长阈值（最大周长的0.1倍）
    double minPerimeter = 0.2 * maxPerimeter;
    // 方法2：旋转矩形
    for (size_t i = 0; i < contours.size(); i++) {
        // 获取最小外接矩形
        RotatedRect rotatedRect = minAreaRect(contours[i]);
        double area = contourArea(contours[i]);

        if (area < minArea) {
            continue;
        }
        double perimeter = cv::arcLength(contours[i], true);
        if (perimeter < minPerimeter) { // minPerimeter是最小周长阈值
            continue;
        }

        // 将矩形存储到向量中
        detectedRects.push_back(rotatedRect);

        // 提取矩形的四个顶点
        Point2f vertices[4];
        rotatedRect.points(vertices);

        // 绘制旋转矩形（使用红色）
        // 注意：如果gray是单通道灰度图，Scalar只需一个参数；如果是BGR图则需要三个参数(0,0,255)
        for (int j = 0; j < 4; j++) {
            // 单通道灰度图用红色（255）
            line(image, vertices[j], vertices[(j + 1) % 4], Scalar(255), 1);

            // 如果是BGR彩色图，使用下面这行
            // line(gray, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 1);
        }
    }
    // 1. 计算平均长度和宽度
    float totalLength = 0, totalWidth = 0;
    int validCount = 0;

    for (const auto& rect : detectedRects) {
        float length, width, angle;
        getLengthAndWidth(rect, length, width, angle);

        totalLength += length;
        totalWidth += width;
        validCount++;
    }
    int validRectCount = validCount;
    float avgLength = totalLength / validCount;
    float avgWidth = totalWidth / validCount;

    // 2. 对矩形进行分组并计算相邻距离
    auto groupedCenters = groupRectanglesByAngle(detectedRects,totalrect.center);

    // 3. 分别处理两组矩形，计算相邻距离
    // 第一组：假设是水平方向（角度接近0度）
    float horizontalAvg = 0;
    float verticalAvg = 0;
    if (!groupedCenters[0].empty()) {
        auto distances = calculateGroupDistances(groupedCenters[0], false); // 按X排序
        // 计算水平组距离的平均值
        
        if (!distances.empty()) {
            float sum = 0;
            for (float d : distances) sum += d;
            horizontalAvg = sum / distances.size();
        }
    }
    // 第二组：假设是垂直方向（角度接近90度）
    if (!groupedCenters[1].empty()) {
        auto distances = calculateGroupDistances(groupedCenters[1], true); // 按Y排序
        // 计算垂直组距离的平均值
        
        if (!distances.empty()) {
            float sum = 0;
            for (float d : distances) sum += d;
            verticalAvg = sum / distances.size();
        }
        
    }
    float avedistances = 0.5 * (verticalAvg + horizontalAvg)- avgWidth;
    // 存储矩形信息
    RectangleInfo info;
    info.width = avgWidth;
    info.height = avgLength;
    info.pitch_inner = avedistances;
    info.pitch_outer = totalrect.size.width - avgLength - 2 * avgLength - avedistances;
    rectInfos.push_back(info);
    return true;
}
bool detectFIF(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea) {
    Mat binary;
    threshold(image, binary, 127, 255, THRESH_BINARY_INV | THRESH_OTSU); // 使用OTSU自动阈值
        //// 4. 形态学操作优化轮廓
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel); // 填充小空洞

    cv::Mat medianFilteredImage;
    cv::medianBlur(image, medianFilteredImage, 3);
    cv::Mat preprocessedImage;
    cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

    // 使用双边滤波减少噪声
    cv::Mat smoothedImage2, edges2;
    cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
    cv::Canny(smoothedImage2, edges2, 170, 200, 3, true);
    // 5. 轮廓检测：使用RETR_CCOMP获取所有层次轮廓（包括内部轮廓）
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // RETR_CCOMP模式可以获取所有外层和内层轮廓
    findContours(edges2, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    //findContours(edges2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //findContours(edges2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    // 使用所有轮廓点计算外接矩形
    vector<Point> allPoints;
    for (const auto& contour : contours) {
        allPoints.insert(allPoints.end(), contour.begin(), contour.end());
    }
    // 使用boundingRect计算外接矩形
    RotatedRect totalrect = minAreaRect(allPoints);
    // 6. 处理轮廓，筛选内部轮廓（重点）
    rectInfos.clear();
    //方法旋转矩形
    // 定义一个向量用于存储检测到的旋转矩形
    std::vector<RotatedRect> detectedRects;
    double maxPerimeter = 0.0;
    double maxarea = 0.0;
    for (size_t i = 0; i < contours.size(); i++) {
        double perimeter = cv::arcLength(contours[i], true);
        double area = contourArea(contours[i]);
        if (perimeter > maxPerimeter) {
            maxPerimeter = perimeter;
        }
        if (area > maxarea)
            maxarea = area;
    }
    for (size_t i = 0; i < contours.size(); i++) {
        // 获取最小外接矩形
        RotatedRect rotatedRect = minAreaRect(contours[i]);
        double area = contourArea(contours[i]);

        if (area < maxarea * 0.05) {
            continue;
        }
        double perimeter = cv::arcLength(contours[i], true);
        if (perimeter < 0.1 * maxPerimeter) { // minPerimeter是最小周长阈值
            continue;
        }

        // 将矩形存储到向量中
        detectedRects.push_back(rotatedRect);

        // 提取矩形的四个顶点
        Point2f vertices[4];
        rotatedRect.points(vertices);

        // 绘制旋转矩形（使用红色）
        // 注意：如果gray是单通道灰度图，Scalar只需一个参数；如果是BGR图则需要三个参数(0,0,255)
        Mat gray_3c;
        // 关键：灰度图 → 3通道BGR图（核心API，直接复制）z
        cvtColor(image, gray_3c, COLOR_GRAY2BGR);
        for (int j = 0; j < 4; j++) {
            // 单通道灰度图用红色（255）
            //line(image, vertices[j], vertices[(j + 1) % 4], Scalar(0), 1);

            // 如果是BGR彩色图，使用下面这行
            line(gray_3c, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 1);
        }
    }
    RectangleInfo info;
    if (detectedRects.size() == 8)
    {
        info.width = 0.5 * (totalrect.size.width + totalrect.size.height);
        info.width_inner = 0.5 * (detectedRects[2].size.width+ detectedRects[2].size.height);
        info.width_outer = 0.5 * (detectedRects[6].size.width + detectedRects[6].size.height);       // 矩形宽度
        info.pitch_inner = 0.5 * (detectedRects[2].size.width + detectedRects[2].size.height) - 0.5 * (detectedRects[0].size.width + detectedRects[0].size.height);
        info.pitch_outer = 0.5 * (detectedRects[6].size.width + detectedRects[6].size.height) - 0.5 * (detectedRects[4].size.width + detectedRects[4].size.height);
        rectInfos.push_back(info);
    }
    else
    {
        return false;
    }
    return true;
}
bool detectBoxinBox(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea) {
    // 3. 二值化处理
    Mat binary;
    threshold(image, binary, 127, 255, THRESH_BINARY_INV | THRESH_OTSU); // 使用OTSU自动阈值

        //// 4. 形态学操作优化轮廓
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel); // 填充小空洞

    cv::Mat medianFilteredImage;
    cv::medianBlur(image, medianFilteredImage, 3);
    cv::Mat preprocessedImage;
    cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

    // 使用双边滤波减少噪声
    cv::Mat smoothedImage2, edges2;
    cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
    cv::Canny(smoothedImage2, edges2, 170, 200, 3, true);
    // 5. 轮廓检测：使用RETR_CCOMP获取所有层次轮廓（包括内部轮廓）
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // RETR_CCOMP模式可以获取所有外层和内层轮廓
    findContours(edges2, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    //findContours(edges2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //findContours(edges2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    // 使用所有轮廓点计算外接矩形
    vector<Point> allPoints;
    for (const auto& contour : contours) {
        allPoints.insert(allPoints.end(), contour.begin(), contour.end());
    }
    // 使用boundingRect计算外接矩形
    RotatedRect totalrect = minAreaRect(allPoints);
    // 6. 处理轮廓，筛选内部轮廓（重点）
    rectInfos.clear();
    //方法旋转矩形
    // 定义一个向量用于存储检测到的旋转矩形
    std::vector<RotatedRect> detectedRects;
    double maxPerimeter = 0.0;
    double maxarea = 0.0;
    for (size_t i = 0; i < contours.size(); i++) {
        double perimeter = cv::arcLength(contours[i], true);
        double area = contourArea(contours[i]);
        if (perimeter > maxPerimeter) {
            maxPerimeter = perimeter;
        }
        if (area > maxarea)
            maxarea = area;
    }
    for (size_t i = 0; i < contours.size(); i++) {
        // 获取最小外接矩形
        RotatedRect rotatedRect = minAreaRect(contours[i]);
        double area = contourArea(contours[i]);

        if (area < maxarea * 0.05) {
            continue;
        }
        double perimeter = cv::arcLength(contours[i], true);
        if (perimeter < 0.1 * maxPerimeter) { // minPerimeter是最小周长阈值
            continue;
        }

        // 将矩形存储到向量中
        detectedRects.push_back(rotatedRect);

        // 提取矩形的四个顶点
        Point2f vertices[4];
        rotatedRect.points(vertices);

        // 绘制旋转矩形（使用红色）
        // 注意：如果gray是单通道灰度图，Scalar只需一个参数；如果是BGR图则需要三个参数(0,0,255)
        Mat gray_3c;
        // 关键：灰度图 → 3通道BGR图（核心API，直接复制）z
        cvtColor(image, gray_3c, COLOR_GRAY2BGR);
        for (int j = 0; j < 4; j++) {
            // 单通道灰度图用红色（255）
            line(image, vertices[j], vertices[(j + 1) % 4], Scalar(0), 1);

            // 如果是BGR彩色图，使用下面这行
            line(gray_3c, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 1);
        }
    }
    RectangleInfo info;
    if (detectedRects.size() == 4)
    {
        info.width = 0.5 * (totalrect.size.width + totalrect.size.height);
        info.width_inner = 0.5 * (detectedRects[0].size.width + detectedRects[0].size.height);
        info.width_outer = 0.5 * (detectedRects[2].size.width + detectedRects[2].size.height);       // 矩形宽度
        rectInfos.push_back(info);
    }
    else
    {
        return false;
    }
    return true;
}
bool detectBarinBar(const Mat& image, vector<RectangleInfo>& rectInfos, int& minArea) {
    // 3. 二值化处理
    Mat binary;
    threshold(image, binary, 127, 255, THRESH_BINARY_INV | THRESH_OTSU); // 使用OTSU自动阈值
    /*adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, 11, 2);*/

        //// 4. 形态学操作优化轮廓
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel); // 填充小空洞

    cv::Mat medianFilteredImage;
    cv::medianBlur(image, medianFilteredImage, 3);
    cv::Mat preprocessedImage;
    cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

    // 使用双边滤波减少噪声
    cv::Mat smoothedImage2, edges2;
    cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
    cv::Canny(smoothedImage2, edges2, 170, 200, 3, true);
    // 5. 轮廓检测：使用RETR_CCOMP获取所有层次轮廓（包括内部轮廓）
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // RETR_EXTERNAL模式可以获取所有外层轮廓
    findContours(edges2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {

        return false;
    }
    // 使用所有轮廓点计算外接矩形
    vector<Point> allPoints;
    for (const auto& contour : contours) {
        allPoints.insert(allPoints.end(), contour.begin(), contour.end());
    }
    // 使用boundingRect计算外接矩形
    RotatedRect totalrect = minAreaRect(allPoints);
    // 6. 处理轮廓，筛选内部轮廓（重点）
    rectInfos.clear();
    //方法旋转矩形
    // 定义一个向量用于存储检测到的旋转矩形
    std::vector<RotatedRect> detectedRects;
    double maxPerimeter = 0.0;
    double maxarea = 0.0;
    for (size_t i = 0; i < contours.size(); i++) {
        double perimeter = cv::arcLength(contours[i], true);
        double area= contourArea(contours[i]);
        if (perimeter > maxPerimeter) {
            maxPerimeter = perimeter;
        }
        if (area > maxarea)
            maxarea = area;
    }
    Mat gray_3c;
    for (size_t i = 0; i < contours.size(); i++) {
        // 获取最小外接矩形
        RotatedRect rotatedRect = minAreaRect(contours[i]);
        double area = contourArea(contours[i]);

        if (area < maxarea*0.1) {
            continue;
        }
        double perimeter = cv::arcLength(contours[i], true);
        if (perimeter < 0.1* maxPerimeter) { // minPerimeter是最小周长阈值
            continue;
        }

        // 将矩形存储到向量中
        detectedRects.push_back(rotatedRect);

        // 提取矩形的四个顶点
        Point2f vertices[4];
        rotatedRect.points(vertices);

        // 绘制旋转矩形（使用红色）
        // 注意：如果gray是单通道灰度图，Scalar只需一个参数；如果是BGR图则需要三个参数(0,0,255)
       
        // 关键：灰度图 → 3通道BGR图（核心API，直接复制）
        cvtColor(image, gray_3c, COLOR_GRAY2BGR);
        for (int j = 0; j < 4; j++) {
            // 如果是BGR彩色图，使用下面这行
            line(gray_3c, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 1);
        }
    }
    // 1. 按面积排序
    vector<pair<float, RotatedRect>> rectsWithArea;
    for (const auto& rect : detectedRects) {
        float area = rect.size.width * rect.size.height;
        rectsWithArea.emplace_back(area, rect);
    }

    sort(rectsWithArea.begin(), rectsWithArea.end(),
        [](const pair<float, RotatedRect>& a, const pair<float, RotatedRect>& b) {
            return a.first < b.first;
        });

    // 2. 分离内矩形和外矩形
    vector<RotatedRect> innerRects;
    vector<RotatedRect> outerRects;
    float innerLength=0.0;
    float outerLength = 0.0;
    float innerWidth = 0.0;
    float outerWidth = 0.0;
    int validCount;
    float innerDistances = 0.0;
    float outerDistances = 0.0;
    if (detectedRects.size() == 8)
    {
        for (int i = 0; i < 4; i++) {
        RotatedRect rect = rectsWithArea[i].second;
        float length, width, angle;
        getLengthAndWidth(rect, length, width, angle);
        innerLength += length;
        innerWidth += width;

        float Dist = calculateDistance(rect.center,totalrect.center);
        innerDistances += Dist;
    }
    
    for (int i = 4; i < 8; i++) {
        RotatedRect rect = rectsWithArea[i].second;
        float length, width, angle;
        getLengthAndWidth(rect, length, width, angle);
        outerLength += length;
        outerWidth += width;

        float Dist = calculateDistance(rect.center,totalrect.center);
        outerDistances += Dist;
    }
    }
    
    innerDistances = innerDistances / 4;
    outerDistances = outerDistances / 4;
    // 存储矩形信息
    RectangleInfo info;
    info.width = 0.5 * (totalrect.size.width + totalrect.size.height);
    info.width_inner = innerWidth / 4;
    info.height_inner = innerLength / 4;
    info.width_outer= outerWidth/4;       // 矩形宽度
    info.height_outer= outerLength/4;      // 矩形高度
    info.pitch_inner = outerDistances- innerDistances;

    rectInfos.push_back(info);
    //按位置排序

    return true;
}
bool detectWaferCrossMarks(const Mat& image, vector<CrossMarkInfo>& crossInfos, int& minFeatureSize) {
    if (image.empty()) {
        return false;
    }

    // ====================== 1. 预处理：增强十字标记特征 ======================
    Mat gray, blurred, edges;
    // 灰度转换
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }

    // 高斯模糊降噪（适配十字细线特征）
    GaussianBlur(gray, blurred, Size(5, 5), 1.5);

    // 自适应二值化（增强对比度，适配不均匀光照）
    Mat binary;
    adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 3);

    // Canny边缘检测（提取十字边缘）
    Canny(binary, edges, 50, 150, 3);

    // ====================== 2. 霍夫直线检测：提取横竖交叉线 ======================
    vector<Vec4i> lines;
    // 检测线段：参数可根据实际十字尺寸调整
    HoughLinesP(edges, lines, 1, CV_PI / 180, 40, 30, 8);

    // 分类水平/垂直线段（按角度筛选）
    vector<Vec4i> horizontalLines, verticalLines;
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double dx = x2 - x1, dy = y2 - y1;
        double angle = abs(atan2(dy, dx) * 180 / CV_PI); // 线段角度

        // 水平线段：角度接近0/180°；垂直线段：角度接近90°
        if (angle < 10 || angle > 170) {
            horizontalLines.push_back(line);
        }
        else if (angle > 80 && angle < 100) {
            verticalLines.push_back(line);
        }
    }

    // ====================== 3. 轮廓检测：补充十字形轮廓验证 ======================
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    crossInfos.clear();

    // ====================== 4. 融合直线+轮廓：提取十字标记尺寸 ======================
    // 方法1：基于霍夫直线的十字交叉分析
    for (const auto& hLine : horizontalLines) {
        for (const auto& vLine : verticalLines) {
            int hx1 = hLine[0], hy1 = hLine[1], hx2 = hLine[2], hy2 = hLine[3];
            int vx1 = vLine[0], vy1 = vLine[1], vx2 = vLine[2], vy2 = vLine[3];

            // 计算十字交叉点
            Point2f crossCenter = getLineIntersection(hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2);
            if (crossCenter.x == -1 && crossCenter.y == -1) continue; // 无有效交点

            // 计算水平臂长度（交叉点到线段两端的距离和）
            double hLen1 = norm(Point2f(hx1, hy1) - crossCenter);
            double hLen2 = norm(Point2f(hx2, hy2) - crossCenter);
            double totalHLen = hLen1 + hLen2;

            // 计算垂直臂长度
            double vLen1 = norm(Point2f(vx1, vy1) - crossCenter);
            double vLen2 = norm(Point2f(vx2, vy2) - crossCenter);
            double totalVLen = vLen1 + vLen2;

            // 筛选最小特征尺寸（避免噪声）
            if (totalHLen < minFeatureSize || totalVLen < minFeatureSize) continue;

            // 计算臂宽度（线段包围矩形的高度/宽度）
            Rect hRect = boundingRect(Mat(vector<Point>{Point(hx1, hy1), Point(hx2, hy2)}));
            Rect vRect = boundingRect(Mat(vector<Point>{Point(vx1, vy1), Point(vx2, vy2)}));
            double hWidth = hRect.height;
            double vWidth = vRect.width;

            // 计算十字包围矩形
            Rect crossRect(
                crossCenter.x - totalHLen / 2, crossCenter.y - totalVLen / 2,
                totalHLen, totalVLen
            );

            // 存储十字标记信息
            CrossMarkInfo info;
            info.center = crossCenter;
            info.horizontalArmLen = totalHLen;
            info.verticalArmLen = totalVLen;
            info.horizontalArmWidth = hWidth;
            info.verticalArmWidth = vWidth;
            info.boundingRect = crossRect;

            crossInfos.push_back(info);
        }
    }

    // 方法2：基于十字形轮廓的验证（凸缺陷特征）
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < minFeatureSize * minFeatureSize) continue; // 面积筛选

        // 十字形特征：凸缺陷数量为4（四个臂的凹陷）
        vector<Vec4i> defects;
        vector<Point> hull;
        convexHull(contour, hull);
        convexityDefects(contour, hull, defects);
        if (defects.size() != 4) continue;

        // 计算轮廓中心（矩中心）
        Moments mu = moments(contour);
        Point2f center(mu.m10 / mu.m00, mu.m01 / mu.m00);

        // 计算四个臂的长度（凸缺陷点到中心的距离）
        vector<double> armLens;
        for (const auto& defect : defects) {
            int defectIdx = defect[2];
            Point2f defectPt = contour[defectIdx];
            armLens.push_back(norm(defectPt - center));
        }

        // 水平/垂直臂平均长度
        double hLen = (armLens[0] + armLens[2]) * 2;
        double vLen = (armLens[1] + armLens[3]) * 2;

        // 存储轮廓提取的十字信息
        CrossMarkInfo info;
        info.center = center;
        info.horizontalArmLen = hLen;
        info.verticalArmLen = vLen;
        info.boundingRect = boundingRect(contour);
        info.horizontalArmWidth = info.boundingRect.height;
        info.verticalArmWidth = info.boundingRect.width;

        crossInfos.push_back(info);
    }

    if (crossInfos.empty()) {
        return false;
    }

    return true;
}

Point2f getLineIntersection(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4) {
    double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (fabs(denom) < 1e-6) return Point2f(-1, -1); // 平行/重合，无交点

    double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    double u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

    // 交点在线段范围内
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        return Point2f(x1 + t * (x2 - x1), y1 + t * (y2 - y1));
    }
    return Point2f(-1, -1);
}


// 获取矩形的长边和短边
void getLengthAndWidth(const RotatedRect& rect, float& height, float& width, float& angle) {
    // RotatedRect的size中，width不总是大于height
    if (rect.size.width >= rect.size.height) {
        height = rect.size.width;
        width = rect.size.height;
        angle =0;
    }
    else {
        // 如果width < height，交换并调整角度
        height = rect.size.height;
        width = rect.size.width;
        angle =90;
    }
}
vector<vector<Point2f>> groupRectanglesByAngle(const vector<RotatedRect>& rects,Point2f&totalrect) {

    vector<vector<Point2f>> groupedCenters(2); // 两组：0=水平方向，1=垂直方向

    if (rects.empty()) return groupedCenters;

    // 临时存储矩形信息和角度
    vector<pair<Point2f, float>> rectInfoAngles;
    for (const auto& rect : rects) {
        float length, width, angle;
        getLengthAndWidth(rect, length, width, angle);
        rectInfoAngles.emplace_back(rect.center, angle);

        if ((rect.center.x < totalrect.x && rect.center.y < totalrect.y) || (rect.center.x > totalrect.x && rect.center.y > totalrect.y))
        {
            groupedCenters[0].push_back(rect.center);
        }
        else
        {
            groupedCenters[1].push_back(rect.center);
        }
    }

    //// 按角度排序
    //sort(rectInfoAngles.begin(), rectInfoAngles.end(),
    //    [](const pair<Point2f, float>& a, const pair<Point2f, float>& b) {
    //        return a.second < b.second;
    //    });

    //// 简单聚类分组
    //if (rectInfoAngles.size() >= 2) {
    //    // 使用第一个矩形的角度作为第一组参考
    //    float refAngle1 = rectInfoAngles[0].second;

    //    for (const auto& [center, angle] : rectInfoAngles) {
    //        float angleDiff = abs(angle - refAngle1);

    //        // 如果角度接近参考角度，归为第一组，否则归为第二组
    //        if (angleDiff ==0) {
    //            groupedCenters[0].push_back(center);
    //        }
    //        else {
    //            groupedCenters[1].push_back(center);
    //        }
    //    }
    //}
    //else {
    //    // 只有一个矩形的情况
    //    groupedCenters[0].push_back(rectInfoAngles[0].first);
    //}

    return groupedCenters;
}
// 计算矩形组的相邻距离
vector<float> calculateGroupDistances(const vector<Point2f>& centers, bool sortByX) {
    vector<float> distances;

    if (centers.size() < 2) return distances;

    // 复制中心点以便排序
    vector<Point2f> sortedCenters = centers;

    // 根据方向排序
    if (sortByX) {
        // 水平方向：按X坐标排序
        sort(sortedCenters.begin(), sortedCenters.end(),
            [](const Point2f& a, const Point2f& b) {
                return a.x < b.x;
            });
    }
    else {
        // 垂直方向：按Y坐标排序
        sort(sortedCenters.begin(), sortedCenters.end(),
            [](const Point2f& a, const Point2f& b) {
                return a.y < b.y;
            });
    }
    if (sortedCenters.size() == 4) {
        // 情况1：有4个点，计算前2个点的距离和后2个点的距离
        float dist1 = norm(sortedCenters[0] - sortedCenters[1]);
        float dist2 = norm(sortedCenters[2] - sortedCenters[3]);
        distances.push_back(dist1);
        distances.push_back(dist2);

        // 可选：如果还需要计算两组中心点之间的距离（如中间两个点1和2）
        // float dist_between_groups = norm(sortedCenters[1] - sortedCenters[2]);
        // distances.push_back(dist_between_groups);
    }
    else {
        // 情况2：其他数量的点，计算所有相邻点的距离
        for (size_t i = 0; i < sortedCenters.size() - 1; i++) {
            float dist = norm(sortedCenters[i] - sortedCenters[i + 1]);
            distances.push_back(dist);
        }
    }

    return distances;
}
// 计算中心点距离
float calculateDistance(const Point2f& p1, const Point2f& p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}