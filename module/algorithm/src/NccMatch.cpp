#include "NccMatch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // 通常和图像处理函数一起包含
#include "StegerSubpixel.h"
#include "icp.h"
#include "core/core.hpp"


#define DRAW_MATCH_EDGE 1
#define CV_PI_RAD (CV_PI / 180.0f)
NccMatch::NccMatch(int angleStep, int minAngle, int maxAngle,
    double cannyThresh1, double cannyThresh2,
    double contourAreaThresh)
    : angleStep_(angleStep), minAngle_(minAngle), maxAngle_(maxAngle),
    cannyThresh1_(cannyThresh1), cannyThresh2_(cannyThresh2),
    cannyApertureSize_(3), cannyL2gradient_(true),
    contourAreaThresh_(contourAreaThresh) {}

void NccMatch::setAngleParams(int angleStep, int minAngle, int maxAngle) {
    angleStep_ = angleStep;
    minAngle_ = minAngle;
    maxAngle_ = maxAngle;
}

void NccMatch::setCannyParams(double thresh1, double thresh2, int apertureSize, bool L2gradient) {
    cannyThresh1_ = thresh1;
    cannyThresh2_ = thresh2;
    cannyApertureSize_ = apertureSize;
    cannyL2gradient_ = L2gradient;
}

void NccMatch::setContourAreaThreshold(double threshold) {
    contourAreaThresh_ = threshold;
}

bool NccMatch::extractEdgePoints(const cv::Mat& grayImage, std::vector<cv::Point>& edgePoints) {
    // 边缘检测
    cv::Mat edges;
    //cv::Canny(grayImage, edges, cannyThresh1_, cannyThresh2_, cannyApertureSize_, cannyL2gradient_);
    cv::Canny(grayImage, edges, 190, 230, 3, true);
    lastEdgeImage_ = edges.clone();
    int area = 20;
    // 检测外层轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(edges.clone(), contours, hierarchy,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 创建与edges相同尺寸的黑色图像
    cv::Mat contourImage = cv::Mat::zeros(edges.size(), CV_8UC1);

    // 绘制所有外层轮廓为白色（255表示白色）
    cv::drawContours(contourImage, contours, -1, cv::Scalar(255), 1);
    edges = contourImage.clone();
    bool has_no_edge = (countNonZero(edges) == 0);
    if (has_no_edge==0)
        extractEdgePointsWithNoiseFilter(edges, edgePoints, area);
    if (edgePoints.empty())
    {
        // 图像预处理增强
         // 预处理：中值滤波去除椒盐噪声
        cv::Mat medianFilteredImage;
        cv::medianBlur(grayImage, medianFilteredImage, 3);
        cv::Mat preprocessedImage;
        cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

        // 使用双边滤波减少噪声
        cv::Mat smoothedImage2, edges2;
        cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
        //cv::Canny(smoothedImage2, edges2, 200, 230, 3, true);
        cv::Canny(smoothedImage2, edges2, 170, 230, 3, true);
       area = 250;
        extractEdgePointsWithNoiseFilter(edges2, edgePoints, area);
    }
    if(edgePoints.size()>5000)
        edgePoints = sparseEdgePointsSimple(edgePoints, 3.0f);
    //// 提取轮廓并过滤小面积轮廓
    //std::vector<std::vector<cv::Point>> contours;
    //cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //edgePoints.clear();
    //for (const auto& contour : contours) {
    //    if (cv::contourArea(contour) > contourAreaThresh_) {
    //        edgePoints.insert(edgePoints.end(), contour.begin(), contour.end());
    //    }
    //}

    return !edgePoints.empty();
}
std::vector<cv::Point> NccMatch::sparseEdgePointsSimple(
    const vector<Point>& srcEdgePoints,
    float minDistThreshold)
{
    std::vector<cv::Point> dstEdgePoints;
    if (srcEdgePoints.empty() || minDistThreshold <= 0) return dstEdgePoints;

    const float cellSize = minDistThreshold;
    const float distSqThreshold = minDistThreshold * minDistThreshold;

    // 创建网格
    std::unordered_map<int, std::vector<int>> gridMap; // grid key -> point indices

    // 预分配空间
    dstEdgePoints.reserve(srcEdgePoints.size() / 4); // 预估稀疏化后的大小

    for (size_t i = 0; i < srcEdgePoints.size(); ++i) {
        const cv::Point& currPt = srcEdgePoints[i];

        // 计算当前点所在的网格坐标
        int gridX = static_cast<int>(currPt.x / cellSize);
        int gridY = static_cast<int>(currPt.y / cellSize);

        // 检查相邻网格中的点
        bool isValid = true;

        // 检查3x3的相邻网格
        for (int dx = -1; dx <= 1 && isValid; ++dx) {
            for (int dy = -1; dy <= 1 && isValid; ++dy) {
                int key = (gridX + dx) * 10007 + (gridY + dy); // 简单的哈希
                auto it = gridMap.find(key);
                if (it != gridMap.end()) {
                    for (int idx : it->second) {
                        const cv::Point& savedPt = dstEdgePoints[idx];
                        float dx = currPt.x - savedPt.x;
                        float dy = currPt.y - savedPt.y;
                        if (dx * dx + dy * dy <= distSqThreshold) {
                            isValid = false;
                            break;
                        }
                    }
                }
            }
        }

        if (isValid) {
            dstEdgePoints.push_back(currPt);
            int key = gridX * 10007 + gridY;
            gridMap[key].push_back(dstEdgePoints.size() - 1);
        }
    }

    return dstEdgePoints;
}
cv::Mat NccMatch::computeFourierTemplate(const cv::Mat& spatialTemplate) {
    if (spatialTemplate.empty()) return cv::Mat();

    // 计算最优DFT尺寸
    int m = cv::getOptimalDFTSize(spatialTemplate.rows);
    int n = cv::getOptimalDFTSize(spatialTemplate.cols);

    // 填充图像到最优尺寸
    cv::Mat padded;
    cv::copyMakeBorder(spatialTemplate, padded, 0, m - spatialTemplate.rows,
        0, n - spatialTemplate.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 执行DFT
    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);

    // 计算幅度谱并中心化
    cv::split(complex, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];

    // 对数变换增强低幅度值
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);

    // 频谱中心化
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;
    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(q0.clone()); q3.copyTo(q0); q0.clone().copyTo(q3);
    q1.copyTo(q1.clone()); q2.copyTo(q1); q1.clone().copyTo(q2);

    // 归一化
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return mag;
}

bool NccMatch::generateTemplates(const std::vector<cv::Point>& edgePoints,
    std::vector<cv::Point2f>& subedgePoints,
    const cv::Mat& refImage) {
    auto start = std::chrono::high_resolution_clock::now();
    // 清空所有模板容器
    spatialTemplates_.clear();
    fourierTemplates_.clear();
    template_pyrdown_.clear();
    rotatedSubedgePoints_.clear();  // 新增：存储各角度旋转后的亚像素点（与模板角度对应）

    // 基础校验：确保像素点与亚像素点数量一致
    if (edgePoints.empty() || subedgePoints.empty() || edgePoints.size() != subedgePoints.size()) {

        return false;
    }

    // -------------------------- 步骤1：生成0层原始模板（含亚像素点） --------------------------
    baseBbox_ = cv::boundingRect(edgePoints);
    int baseW = baseBbox_.width;
    int baseH = baseBbox_.height;
    if (baseW < 20 || baseH < 20) {
        return false;
    }

    // 创建0层基准模板（像素点+亚像素点）
    cv::Mat baseTemp = cv::Mat::zeros(baseH, baseW, CV_8UC1);

    // 1.1 绘制像素点
    for (const auto& p : edgePoints) {
        int x = p.x - baseBbox_.x;
        int y = p.y - baseBbox_.y;
        if (x >= 0 && x < baseW && y >= 0 && y < baseH) {
            baseTemp.at<uchar>(y, x) = 255;
        }
    }
    // 计算图像中心坐标（浮点数，确保精度，尤其是偶数尺寸图像）
    float cx = (baseTemp.cols - 1) / 2.0f;  // 图像宽度方向中心（x轴中心）
    float cy = (baseTemp.rows - 1) / 2.0f;  // 图像高度方向中心（y轴中心）

    // 1.2 绘制亚像素点（原始位置）并保存原始亚像素映射坐标
    std::vector<cv::Point2f> baseSubedgePoints;  // 基于baseBbox_的亚像素坐标
    baseSubedgePoints.reserve(subedgePoints.size());
    for (const auto& p : subedgePoints) {
        double x = p.x - baseBbox_.x;  // 与像素点相同的基准映射
        double y = p.y - baseBbox_.y;
        if (x >= 0 && x < baseW && y >= 0 && y < baseH) {
            cv::circle(baseTemp, cv::Point2f(x, y), 0.5, cv::Scalar(255), -1, cv::LINE_AA);
            baseSubedgePoints.emplace_back(x, y);  // 保存映射后的原始亚像素点
        }
    }
    // 将以图像中心为原点的亚像素点存入template_features.txt（仅亚像素点）
    std::ofstream outFile("template_features.txt");
    if (!outFile.is_open()) {
        return false;  // 直接返回错误，避免后续无效操作
    }

    // 遍历亚像素点，转换为中心坐标并写入
    for (const auto& subPt : baseSubedgePoints) {
        // 亚像素点以图像中心为原点的坐标：x = 亚像素x - 中心x；y = 亚像素y - 中心y
        float centerX = subPt.x - cx;
        float centerY = subPt.y - cy;
        // 写入转换后的亚像素坐标（保留小数精度）
        outFile << centerX << " " << centerY << std::endl;
    }

    outFile.close();  // 关闭文件流
    // -------------------------- 步骤2：生成2层压缩模板（像素点逻辑不变） --------------------------
    cv::Point2f baseCenter((baseBbox_.x + baseBbox_.width / 2.0f),
        (baseBbox_.y + baseBbox_.height / 2.0f));

    std::vector<cv::Point> scaled1EdgePoints, scaled2EdgePoints;
    // 1层压缩
    for (const auto& p : edgePoints) {
        float dx1 = (p.x - baseCenter.x) * 0.5f;
        float dy1 = (p.y - baseCenter.y) * 0.5f;
        scaled1EdgePoints.emplace_back(cvRound(baseCenter.x + dx1), cvRound(baseCenter.y + dy1));
    }
    // 2层压缩
    for (const auto& p : scaled1EdgePoints) {
        float dx2 = (p.x - baseCenter.x) * 0.5f;
        float dy2 = (p.y - baseCenter.y) * 0.5f;
        scaled2EdgePoints.emplace_back(cvRound(baseCenter.x + dx2), cvRound(baseCenter.y + dy2));
    }

    cv::Rect scaled2Bbox = cv::boundingRect(scaled2EdgePoints);
    int scaled2W = scaled2Bbox.width;
    int scaled2H = scaled2Bbox.height;
    if (scaled2W < 5 || scaled2H < 5) {
        return false;
    }

    cv::Mat scaled2BaseTemp = cv::Mat::zeros(scaled2H, scaled2W, CV_8UC1);
    for (const auto& p : scaled2EdgePoints) {
        int x = p.x - scaled2Bbox.x;
        int y = p.y - scaled2Bbox.y;
        if (x >= 0 && x < scaled2W && y >= 0 && y < scaled2H) {
            scaled2BaseTemp.at<uchar>(y, x) = 255;
        }
    }

    // -------------------------- 步骤3：生成多角度模板（核心：同步旋转亚像素点） --------------------------
    std::vector<cv::Mat> tempSpatialTemps;
    cv::Point2f origTempCenter(baseW / 2.0f, baseH / 2.0f);  // 0层模板中心（与像素点旋转中心一致）
    cv::Point2f scaled2TempCenter(scaled2W / 2.0f, scaled2H / 2.0f);

    for (int angle = minAngle_; angle <= maxAngle_; angle += angleStep_) {
        float angleRad = static_cast<float>(angle * CV_PI / 180.0);

        // 3.1 生成0层旋转模板（改为：直接对边缘点做变换，而非图像变换）
        cv::RotatedRect origRotRect(origTempCenter, cv::Size(baseW, baseH), angle);
        cv::Rect origRotBbox = origRotRect.boundingRect();
        cv::Mat origRotMat = cv::getRotationMatrix2D(origTempCenter, angle, 1.0);
        // 平移校正（保持与原逻辑完全一致，确保坐标偏移同步）
        origRotMat.at<double>(0, 2) += (origRotBbox.width / 2.0 - origTempCenter.x);
        origRotMat.at<double>(1, 2) += (origRotBbox.height / 2.0 - origTempCenter.y);

        // -------------------------- 核心修改：点变换替代图像变换 --------------------------
        // 1. 获取baseTemp中的原始边缘点（像素点，确保无冗余）
        // （若已提前提取过baseEdgePoints，直接用；未提取则临时提取一次）
        std::vector<cv::Point> baseEdgePoints;
        if (baseEdgePoints.empty()) { // 避免重复提取，提高效率
            for (int y = 0; y < baseTemp.rows; ++y) {
                const uchar* rowPtr = baseTemp.ptr<uchar>(y);
                for (int x = 0; x < baseTemp.cols; ++x) {
                    if (rowPtr[x] == 255) {
                        baseEdgePoints.push_back(cv::Point(x, y));
                    }
                }
            }
        }

        // 2. 对原始边缘点应用仿射变换（与图像变换用同一矩阵，确保一致性）
        std::vector<cv::Point> rotatedEdgePoints;
        rotatedEdgePoints.reserve(baseEdgePoints.size()); // 预分配空间，提升效率
        for (const auto& pt : baseEdgePoints) {
            // 像素点仿射变换公式（和亚像素点变换逻辑一致）
            cv::Point rotatedPt;
            rotatedPt.x = static_cast<int>(
                origRotMat.at<double>(0, 0) * pt.x +
                origRotMat.at<double>(0, 1) * pt.y +
                origRotMat.at<double>(0, 2) + 0.5 // +0.5是四舍五入，确保像素坐标整数精度
                );
            rotatedPt.y = static_cast<int>(
                origRotMat.at<double>(1, 0) * pt.x +
                origRotMat.at<double>(1, 1) * pt.y +
                origRotMat.at<double>(1, 2) + 0.5
                );
            // 校验是否在旋转后的边界框内（有效点才保留）
            if (rotatedPt.x >= 0 && rotatedPt.x < origRotBbox.width &&
                rotatedPt.y >= 0 && rotatedPt.y < origRotBbox.height) {
                rotatedEdgePoints.push_back(rotatedPt);
            }
        }

        // 3. 基于变换后的边缘点，重新绘制旋转模板（无冗余边缘点）
        cv::Mat origRotatedTemp = cv::Mat::zeros(origRotBbox.size(), CV_8UC1);
        for (const auto& pt : rotatedEdgePoints) {
            origRotatedTemp.at<uchar>(pt.y, pt.x) = 255; // 仅绘制有效变换点
        }
        // --------------------------------------------------------------------------------

        // 3.2 同步旋转亚像素点（核心逻辑不变，保持与像素点变换同步）
        std::vector<cv::Point2f> rotatedSubPoints;
        rotatedSubPoints.reserve(baseSubedgePoints.size());
        for (const auto& subPt : baseSubedgePoints) {
            cv::Point2f rotatedPt;
            rotatedPt.x = static_cast<float>(
                origRotMat.at<double>(0, 0) * subPt.x +
                origRotMat.at<double>(0, 1) * subPt.y +
                origRotMat.at<double>(0, 2)
                );
            rotatedPt.y = static_cast<float>(
                origRotMat.at<double>(1, 0) * subPt.x +
                origRotMat.at<double>(1, 1) * subPt.y +
                origRotMat.at<double>(1, 2)
                );
            if (rotatedPt.x >= 0 && rotatedPt.x < origRotBbox.width &&
                rotatedPt.y >= 0 && rotatedPt.y < origRotBbox.height) {
                rotatedSubPoints.push_back(rotatedPt);
            }
        }
        rotatedSubedgePoints_.emplace_back(rotatedSubPoints, angle);

        // 3.3 生成2层旋转模板（像素点逻辑不变，若需优化可参考0层改为点变换）
        cv::RotatedRect scaled2RotRect(scaled2TempCenter, cv::Size(scaled2W, scaled2H), angle);
        cv::Rect scaled2RotBbox = scaled2RotRect.boundingRect();
        cv::Mat scaled2RotMat = cv::getRotationMatrix2D(scaled2TempCenter, angle, 1.0);
        scaled2RotMat.at<double>(0, 2) += (scaled2RotBbox.width / 2.0 - scaled2TempCenter.x);
        scaled2RotMat.at<double>(1, 2) += (scaled2RotBbox.height / 2.0 - scaled2TempCenter.y);

        cv::Mat scaled2RotatedTemp;
        cv::warpAffine(scaled2BaseTemp, scaled2RotatedTemp, scaled2RotMat, scaled2RotBbox.size(),
            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        tempSpatialTemps.push_back(scaled2RotatedTemp);

        // 角度为0时，直接用原始baseTemp（避免变换误差）
        if (angle == 0) {
            origRotatedTemp = baseTemp.clone();
            origRotBbox = cv::Rect(0, 0, baseTemp.cols, baseTemp.rows); // 重置边界框为原始尺寸
        }

        // 3.4 存储模板（逻辑不变）
        std::vector<cv::Mat> layers;
        layers.push_back(origRotatedTemp.clone());
        layers.push_back(scaled2RotatedTemp.clone());
        template_pyrdown_.emplace_back(layers, angleRad);
    }

    // -------------------------- 步骤4：模板对齐与傅里叶变换（保持不变） --------------------------
    int maxRows = 0, maxCols = 0;
    for (const auto& templ : tempSpatialTemps) {
        maxRows = std::max(maxRows, templ.rows);
        maxCols = std::max(maxCols, templ.cols);
    }
    maxTemplateSize_ = cv::Size(maxCols, maxRows);

    std::vector<cv::Mat> alignedTemplates;
    for (const auto& templ : tempSpatialTemps) {
        cv::Mat alignedTemp = cv::Mat::zeros(maxTemplateSize_, templ.type());
        int rowOffset = cvRound(static_cast<double>(maxRows - templ.rows) / 2.0);
        int colOffset = cvRound(static_cast<double>(maxCols - templ.cols) / 2.0);
        templ.copyTo(alignedTemp(cv::Rect(colOffset, rowOffset, templ.cols, templ.rows)));
        alignedTemplates.push_back(alignedTemp);
    }

    downsampledRefImage_ = refImage.clone();
    for (int j = 1; j < 4; j *= 2) {
        cv::Mat tempDown;
        cv::pyrDown(downsampledRefImage_, tempDown,
            cv::Size(downsampledRefImage_.cols / 2, downsampledRefImage_.rows / 2));
        downsampledRefImage_ = tempDown;
    }

    corrSize_ = cv::Size(
        downsampledRefImage_.cols - maxTemplateSize_.width + 1,
        downsampledRefImage_.rows - maxTemplateSize_.height + 1
    );

    const int ctype = CV_32F;
    for (size_t i = 0; i < alignedTemplates.size(); ++i) {
        const auto& templ = alignedTemplates[i];
        cv::Mat dftTempl;
        dftTemp(downsampledRefImage_, templ, dftTempl, ctype);  // 假设dftTempl是正确的函数名
        if (dftTempl.empty()) {
            continue;
        }

        spatialTemplates_.emplace_back(templ, template_pyrdown_[i].second);
        fourierTemplates_.push_back(dftTempl);
    }

    // -------------------------- 结果校验 --------------------------
    bool success = !spatialTemplates_.empty() && !fourierTemplates_.empty()
        && !template_pyrdown_.empty() && !rotatedSubedgePoints_.empty();

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return success;
}
//优化生成模板时间
bool NccMatch::generateTemplates1(const std::vector<cv::Point>& edgePoints,
    std::vector<cv::Point2f>& subedgePoints,
    const cv::Mat& refImage) {
    auto start = std::chrono::high_resolution_clock::now();
    // 清空所有模板容器
    spatialTemplates_.clear();
    fourierTemplates_.clear();
    template_pyrdown_.clear();
    rotatedSubedgePoints_.clear();

    // 基础校验
    if (edgePoints.empty() || subedgePoints.empty() || edgePoints.size() != subedgePoints.size()) {
        return false;
    }

    // 预计算角度数量
    int nAngles = (maxAngle_ - minAngle_) / angleStep_ + 1;

    // 预分配内存
    template_pyrdown_.reserve(nAngles);
    rotatedSubedgePoints_.reserve(nAngles);

    // -------------------------- 步骤1：生成0层原始模板 --------------------------
    baseBbox_ = cv::boundingRect(edgePoints);
    int baseW = baseBbox_.width;
    int baseH = baseBbox_.height;
    if (baseW < 20 || baseH < 20) {
        return false;
    }

    // 计算图像中心坐标
    float cx = (baseW) * 0.5f;
    float cy = (baseH) * 0.5f;

    // 预计算baseSubedgePoints
    std::vector<cv::Point2f> baseSubedgePoints;
    baseSubedgePoints.reserve(subedgePoints.size());

    // 预计算变换到基坐标系的像素点
    std::vector<cv::Point> baseEdgePoints;
    baseEdgePoints.reserve(edgePoints.size());

    for (size_t i = 0; i < edgePoints.size(); i++) {
        // 像素点
        int px = edgePoints[i].x - baseBbox_.x;
        int py = edgePoints[i].y - baseBbox_.y;
        if (px >= 0 && px < baseW && py >= 0 && py < baseH) {
            baseEdgePoints.emplace_back(px, py);
        }

        // 亚像素点
        float sx = subedgePoints[i].x - baseBbox_.x;
        float sy = subedgePoints[i].y - baseBbox_.y;
        if (sx >= 0 && sx < baseW && sy >= 0 && sy < baseH) {
            baseSubedgePoints.emplace_back(sx, sy);
        }
    }

    // 创建baseTemp并绘制像素点（避免重复检查边界）
    cv::Mat baseTemp = cv::Mat::zeros(baseH, baseW, CV_8UC1);
    for (const auto& p : baseEdgePoints) {
        baseTemp.ptr<uchar>(p.y)[p.x] = 255;
    }

    //// 保存特征点（只在需要时）
    //std::ofstream outFile("template_features.txt");
    //if (outFile.is_open()) {
    //    for (const auto& subPt : baseSubedgePoints) {
    //        outFile << (subPt.x - cx+0.5) << " " << (subPt.y - cy+0.5) << "\n";
    //    }
    //    outFile.close();
    //}
    
    //sm::Core::get_init();
    
    sm::Core* core = sm::Core::get_init();
    // 2. 清空 temp_features（如果需要）
    core->temp_features.clear();

    // 3. 将点存入 temp_features
    for (const auto& subPt : baseSubedgePoints) {
        Point_f pt;
        pt.x = subPt.x - cx + 0.5;
        pt.y = subPt.y - cy + 0.5;
        core->temp_features.push_back(pt);
    }
    // -------------------------- 步骤2：预计算压缩模板 --------------------------
    cv::Point2f baseCenter(baseBbox_.x + baseBbox_.width * 0.5f,
        baseBbox_.y + baseBbox_.height * 0.5f);

    // 一次性计算2层压缩点
    std::vector<cv::Point> scaled2EdgePoints;
    scaled2EdgePoints.reserve(edgePoints.size());

    for (const auto& p : edgePoints) {
        // 直接计算2层压缩（避免中间存储）
        float dx = (p.x - baseCenter.x) * 0.25f;  // 0.5 * 0.5 = 0.25
        float dy = (p.y - baseCenter.y) * 0.25f;
        scaled2EdgePoints.emplace_back(
            cvRound(baseCenter.x + dx),
            cvRound(baseCenter.y + dy)
        );
    }

    cv::Rect scaled2Bbox = cv::boundingRect(scaled2EdgePoints);
    int scaled2W = scaled2Bbox.width;
    int scaled2H = scaled2Bbox.height;
    if (scaled2W < 5 || scaled2H < 5) {
        return false;
    }

    // 创建2层模板
    cv::Mat scaled2BaseTemp = cv::Mat::zeros(scaled2H, scaled2W, CV_8UC1);
    for (const auto& p : scaled2EdgePoints) {
        int x = p.x - scaled2Bbox.x;
        int y = p.y - scaled2Bbox.y;
        scaled2BaseTemp.ptr<uchar>(y)[x] = 255;
    }

    // 预计算中心点
    cv::Point2f origTempCenter(cx, cy);
    cv::Point2f scaled2TempCenter(scaled2W * 0.5f, scaled2H * 0.5f);

    // 预计算2层旋转矩阵（减少重复计算）
    std::vector<cv::Mat> scaled2RotMats;
    std::vector<cv::Rect> scaled2RotBboxes;
    scaled2RotMats.reserve(nAngles);
    scaled2RotBboxes.reserve(nAngles);

    for (int angle = minAngle_; angle <= maxAngle_; angle += angleStep_) {
        // 计算2层旋转矩阵
        cv::RotatedRect scaled2RotRect(scaled2TempCenter, cv::Size(scaled2W, scaled2H), angle);
        cv::Rect bbox = scaled2RotRect.boundingRect();
        cv::Mat rotMat = cv::getRotationMatrix2D(scaled2TempCenter, angle, 1.0);
        rotMat.at<double>(0, 2) += (bbox.width * 0.5 - scaled2TempCenter.x);
        rotMat.at<double>(1, 2) += (bbox.height * 0.5 - scaled2TempCenter.y);

        scaled2RotMats.push_back(rotMat);
        scaled2RotBboxes.push_back(bbox);
    }

    // -------------------------- 步骤3：并行生成多角度模板 --------------------------
    std::vector<cv::Mat> tempSpatialTemps(nAngles);

     //使用OpenMP并行处理（如果支持）
#pragma omp parallel for schedule(dynamic)
    for (int angleIdx = 0; angleIdx < nAngles; angleIdx++) {
        int angle = minAngle_ + angleIdx * angleStep_;
        float angleRad = angle * CV_PI / 180.0f;

        // 3.1 生成0层旋转模板
        cv::RotatedRect origRotRect(origTempCenter, cv::Size(baseW, baseH), angle);
        cv::Rect origRotBbox = origRotRect.boundingRect();
        cv::Mat origRotMat = cv::getRotationMatrix2D(origTempCenter, angle, 1.0);
        origRotMat.at<double>(0, 2) += (origRotBbox.width * 0.5 - origTempCenter.x);
        origRotMat.at<double>(1, 2) += (origRotBbox.height * 0.5 - origTempCenter.y);

        // 旋转像素点
        cv::Mat origRotatedTemp = cv::Mat::zeros(origRotBbox.size(), CV_8UC1);

        // 直接使用预计算的baseEdgePoints
        for (const auto& pt : baseEdgePoints) {
            // 使用矩阵乘法（避免重复计算sin/cos）
            int x = static_cast<int>(
                origRotMat.at<double>(0, 0) * pt.x +
                origRotMat.at<double>(0, 1) * pt.y +
                origRotMat.at<double>(0, 2) + 0.5
                );
            int y = static_cast<int>(
                origRotMat.at<double>(1, 0) * pt.x +
                origRotMat.at<double>(1, 1) * pt.y +
                origRotMat.at<double>(1, 2) + 0.5
                );

            if (x >= 0 && x < origRotBbox.width && y >= 0 && y < origRotBbox.height) {
                origRotatedTemp.at<uchar>(y, x) = 255;
            }
        }

        // 旋转亚像素点
        std::vector<cv::Point2f> rotatedSubPoints;
        rotatedSubPoints.reserve(baseSubedgePoints.size());

        for (const auto& subPt : baseSubedgePoints) {
            float x = static_cast<float>(
                origRotMat.at<double>(0, 0) * subPt.x +
                origRotMat.at<double>(0, 1) * subPt.y +
                origRotMat.at<double>(0, 2)
                );
            float y = static_cast<float>(
                origRotMat.at<double>(1, 0) * subPt.x +
                origRotMat.at<double>(1, 1) * subPt.y +
                origRotMat.at<double>(1, 2)
                );

            if (x >= 0 && x < origRotBbox.width && y >= 0 && y < origRotBbox.height) {
                rotatedSubPoints.emplace_back(x, y);
            }
        }

        // 线程安全地存储结果
#pragma omp critical
        {
            rotatedSubedgePoints_.emplace_back(rotatedSubPoints, angle);
        }

        // 3.2 生成2层旋转模板（使用预计算的旋转矩阵）
        cv::Mat scaled2RotatedTemp;
        cv::warpAffine(scaled2BaseTemp, scaled2RotatedTemp,
            scaled2RotMats[angleIdx], scaled2RotBboxes[angleIdx].size(),
            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        // 角度为0时，使用原始模板（避免变换误差）
        if (angle == 0) {
            origRotatedTemp = baseTemp.clone();
            origRotBbox = cv::Rect(0, 0, baseW, baseH);
        }

        // 存储模板
        std::vector<cv::Mat> layers;
        layers.push_back(origRotatedTemp.clone());
        layers.push_back(scaled2RotatedTemp.clone());

#pragma omp critical
        {
            template_pyrdown_.emplace_back(layers, angleRad);
            tempSpatialTemps[angleIdx] = scaled2RotatedTemp.clone();
        }
    }

    //法2
//#pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
//    for (int angleIdx = 0; angleIdx < nAngles; ++angleIdx) {
//        const int angle = minAngle_ + angleIdx * angleStep_;
//        const float angleRad = angle * CV_PI_RAD; // 预计算常量，避免循环内重复除法
//        const float cosA = cos(angleRad);
//        const float sinA = sin(angleRad);
//        // 3.1 生成0层旋转模板
//        cv::RotatedRect origRotRect(origTempCenter, cv::Size(baseW, baseH), angle);
//        cv::Rect origRotBbox = origRotRect.boundingRect();
//        //cv::Mat origRotMat = cv::getRotationMatrix2D(origTempCenter, angle, 1.0);
//        //origRotMat.at<double>(0, 2) += (origRotBbox.width * 0.5 - origTempCenter.x);
//        //origRotMat.at<double>(1, 2) += (origRotBbox.height * 0.5 - origTempCenter.y);
//
//        cv::Mat origRotatedTemp = cv::Mat::zeros(origRotBbox.size(), CV_8UC1);
//        //const double* rotMatData = origRotMat.ptr<double>(0); // 一次性获取矩阵指针，减少访问开销
//        //const double a00 = rotMatData[0], a01 = rotMatData[1], a02 = rotMatData[2];
//        //const double a10 = origRotMat.ptr<double>(1)[0], a11 = origRotMat.ptr<double>(1)[1], a12 = origRotMat.ptr<double>(1)[2];
//
//        // 旋转像素点：指针访问+预取矩阵系数，极致提速
//        for (const auto& pt : baseEdgePoints) {
//            /*const int x = static_cast<int>(a00 * pt.x + a01 * pt.y + a02 + 0.5f);
//            const int y = static_cast<int>(a10 * pt.x + a11 * pt.y + a12 + 0.5f);*/
//            int x = static_cast<int>((pt.x - cx) * cosA - (pt.y - cy) * sinA + cx + 0.5f);
//            int y = static_cast<int>((pt.x - cx) * sinA + (pt.y - cy) * cosA + cy + 0.5f);
//            if (x >= 0 && x < origRotBbox.width && y >= 0 && y < origRotBbox.height) {
//                origRotatedTemp.ptr<uchar>(y)[x] = 255;
//            }
//        }
//
//        // 旋转亚像素点
//        std::vector<cv::Point2f> rotatedSubPoints;
//        rotatedSubPoints.reserve(baseSubedgePoints.size());
//        for (const auto& subPt : baseSubedgePoints) {
//            //const float x = static_cast<float>(a00 * subPt.x + a01 * subPt.y + a02);
//            //const float y = static_cast<float>(a10 * subPt.x + a11 * subPt.y + a12);
//            float x = (subPt.x - cx) * cosA - (subPt.y - cy) * sinA + cx;
//            float y = (subPt.x - cx) * sinA + (subPt.y - cy) * cosA + cy;
//            if (x >= 0 && x < origRotBbox.width && y >= 0 && y < origRotBbox.height) {
//                rotatedSubPoints.emplace_back(x, y);
//            }
//        }
//
//        //  索引直接赋值，无锁竞争！
//        rotatedSubedgePoints_.emplace_back(rotatedSubPoints, angle);
//        // 3.2 生成2层旋转模板
//        cv::Mat scaled2RotatedTemp;
//        cv::warpAffine(scaled2BaseTemp, scaled2RotatedTemp, scaled2RotMats[angleIdx],
//            scaled2RotBboxes[angleIdx].size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
//
//        // 角度为0时复用原始模板，避免变换误差
//        if (angle == 0) {
//            origRotatedTemp = baseTemp; // 移除clone，直接赋值（Mat浅拷贝，无开销）
//            origRotBbox = cv::Rect(0, 0, baseW, baseH);
//        }
//
//        std::vector<cv::Mat> layers;
//        layers.push_back(origRotatedTemp.clone());
//        layers.push_back(scaled2RotatedTemp.clone());
//        // 存储模板：索引赋值，无锁竞争
//        template_pyrdown_.emplace_back(layers, angleRad);
//        tempSpatialTemps[angleIdx] = scaled2RotatedTemp.clone();
//
//    }
    // -------------------------- 步骤4：模板对齐与傅里叶变换 --------------------------
    // 查找最大尺寸
    int maxRows = 0, maxCols = 0;
    for (const auto& templ : tempSpatialTemps) {
        maxRows = std::max(maxRows, templ.rows);
        maxCols = std::max(maxCols, templ.cols);
    }
    maxTemplateSize_ = cv::Size(maxCols, maxRows);
    if (maxTemplateSize_.area() == 0) return false;
    // 预分配对齐模板空间
    std::vector<cv::Mat> alignedTemplates(nAngles);

    // 对齐模板
    const int rowOffsetBase = (maxRows >> 1); // 右移代替除法，提速
    const int colOffsetBase = (maxCols >> 1);
    for (int i = 0; i < nAngles; ++i) {
        const auto& templ = tempSpatialTemps[i];
        alignedTemplates[i] = cv::Mat::zeros(maxTemplateSize_, templ.type());
        const int rowOffset = rowOffsetBase - (templ.rows >> 1);
        const int colOffset = colOffsetBase - (templ.cols >> 1);
        templ.copyTo(alignedTemplates[i](cv::Rect(colOffset, rowOffset, templ.cols, templ.rows)));
    }

    // 预计算下采样参考图像（只计算一次）
    downsampledRefImage_ = refImage.clone();
    for (int j = 1; j < 4; j *= 2) {
        cv::Mat tempDown;
        cv::pyrDown(downsampledRefImage_, tempDown,
            cv::Size(downsampledRefImage_.cols / 2, downsampledRefImage_.rows / 2));
        downsampledRefImage_ = tempDown;
    }

    corrSize_ = cv::Size(
        downsampledRefImage_.cols - maxTemplateSize_.width + 1,
        downsampledRefImage_.rows - maxTemplateSize_.height + 1
    );

    // 预分配傅里叶模板空间
    fourierTemplates_.reserve(alignedTemplates.size());
    spatialTemplates_.reserve(alignedTemplates.size());

    // 计算傅里叶变换
    const int ctype = CV_32F;
    for (size_t i = 0; i < alignedTemplates.size(); ++i) {
        cv::Mat dftTempl;
        dftTemp(downsampledRefImage_, alignedTemplates[i], dftTempl, ctype);

        if (!dftTempl.empty()) {
            spatialTemplates_.emplace_back(alignedTemplates[i], template_pyrdown_[i].second);
            fourierTemplates_.push_back(dftTempl);
        }
    }

    // -------------------------- 结果校验 --------------------------
    bool success = !spatialTemplates_.empty() && !fourierTemplates_.empty()
        && !template_pyrdown_.empty() && !rotatedSubedgePoints_.empty();
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return success;
}



bool NccMatch::generateTemplateAndSaveEdgePoints(const cv::Mat& grayImage, const std::string& edgeXmlPath) {
    std::vector<cv::Point> edgePoints;
    if (!extractEdgePoints(grayImage, edgePoints)) {
        return false;
    }
    //提取亚像素点
    std::vector<cv::Point2f> subedgePoints;
    SubPixelByZernike(grayImage, edgePoints, subedgePoints);
    // 将亚像素点写入文件
    std::ofstream outFile("generate_features.txt"); // 创建输出文件流，打开目标文件
    if (!outFile.is_open()) { // 检查文件是否成功打开
        std::cerr << "错误：无法打开文件 generate_features.txt 进行写入！" << std::endl;
        return false; // 若打开失败，根据实际场景处理（如返回错误码）
    }

    // 遍历所有亚像素点，逐个写入文件
    for (const cv::Point2f& subPt : subedgePoints) {
        // 每个点的x和y坐标以空格分隔，每行存储一个点
        outFile << subPt.x << " " << subPt.y << std::endl;
    }

    outFile.close(); // 关闭文件流（可选，对象销毁时会自动关闭，但显式关闭更规范）
    // 保存边缘点和亚像素点到XML

    cv::FileStorage fs(edgeXmlPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "无法打开文件：" << edgeXmlPath << std::endl;
        return false;
    }

    // 保存边缘点（用{}替代Map，兼容性更好）
    fs << "edgePoints" << "[";
    for (const auto& pt : edgePoints) {
        fs << "{" << "x" << pt.x << "y" << pt.y << "}";
    }
    fs << "]";

    // 保存亚像素点
    fs << "subedgePoints" << "[";
    for (const auto& pt : subedgePoints) {
        fs << "{" << "x" << pt.x << "y" << pt.y << "}";
    }
    fs << "]";

    fs.release();;
    // 生成模板
    //return generateTemplates(edgePoints);
    return true;
}

//bool NccMatch::loadEdgePointsAndGenerateTemplates(const std::string& edgeXmlPath) {
//    std::vector<cv::Point> edgePoints;
//    cv::FileStorage fs(edgeXmlPath, cv::FileStorage::READ);
//    if (!fs.isOpened()) {
//        std::cerr << "无法打开文件：" << edgeXmlPath << std::endl;
//        return false;
//    }
//    fs["edgePoints"] >> edgePoints;
//    fs.release();
//
//    return generateTemplates(edgePoints);
//}

void NccMatch::dftimg(const cv::Mat& img, cv::Mat& corr, cv::Mat& dftImg) {
    int depth = img.depth();
    maxDepth_ = (depth > CV_8S) ? CV_64F : CV_32F;

    // 计算最优DFT尺寸
    dftSize_.width = cv::getOptimalDFTSize(img.cols);
    dftSize_.height = cv::getOptimalDFTSize(img.rows);

    // 初始化相关矩阵
    corr.create(img.size(), CV_32F);
    dftImg.create(dftSize_, maxDepth_);
    dftImg.setTo(cv::Scalar::all(0));

    // 复制图像到DFT矩阵
    cv::Mat roi(dftImg, cv::Rect(0, 0, img.cols, img.rows));
    img.convertTo(roi, maxDepth_);

    // 执行DFT
    cv::dft(dftImg, dftImg, 0, img.rows);
}
void NccMatch::dftimg(const cv::Mat& img, cv::Mat& corr, cv::Mat& _dftImg, cv::Size& corrsize, cv::Size& dftsize, int ctype, int& maxDepth,
    double delta, int borderType)
{
    int depth = img.depth();
    int cdepth = CV_MAT_DEPTH(ctype)/*, ccn = CV_MAT_CN(ctype)*/;

    corr.create(corrsize, ctype);

    maxDepth = (depth > CV_8S) ? CV_64F : MAX(MAX(CV_32F, depth), cdepth);

    dftsize.width = MAX(cv::getOptimalDFTSize(img.cols), 2);
    dftsize.height = cv::getOptimalDFTSize(img.rows);

    cv::Mat dftImg(dftsize, maxDepth);

    cv::Size wholeSize = img.size();
    cv::Point roiofs(0, 0);
    cv::Mat img0 = img;

    if (!(borderType & cv::BORDER_ISOLATED))
    {
        img.locateROI(wholeSize, roiofs);
        img0.adjustROI(0, 0, 0, 0);
    }
    borderType |= cv::BORDER_ISOLATED;

    // calculate correlation by blocks	
    cv::Mat dst1(dftImg, cv::Rect(0, 0, img0.cols, img0.rows));

    cv::Mat src = img0;
    dftImg = cv::Scalar::all(0);

    if (dst1.data != src.data)
        src.convertTo(dst1, dst1.depth());

    dft(dftImg, dftImg, 0, img.rows);
    _dftImg = dftImg;

}
void NccMatch::dftTemp(const cv::Mat& img, const cv::Mat& _templ, cv::Mat& _dftTempl, int ctype) {
    cv::Mat templ;
    templ = _templ;  // 复制输入模板

    // 注意：强制将depth设为6（对应CV_64F），保留此逻辑
    int depth = 6;  // 替代原代码的 img.depth()，
    int tdepth = templ.depth();
    int cdepth = CV_MAT_DEPTH(ctype);

    // 模板深度转换：确保与参考图深度兼容
    if (depth != tdepth && tdepth != std::max(CV_32F, depth)) {
        _templ.convertTo(templ, std::max(CV_32F, depth));
        tdepth = templ.depth();  // 更新转换后的深度
    }

    // 计算最大深度（避免溢出）
    int maxDepth = 0;
    maxDepth = (depth > CV_8S) ? CV_64F : std::max(std::max(CV_32F, tdepth), cdepth);

    // 计算最佳DFT尺寸（确保至少为2x2）
    cv::Size dftsize;
    dftsize.width = std::max(cv::getOptimalDFTSize(img.cols), 2);
    dftsize.height = std::max(cv::getOptimalDFTSize(img.rows), 2);

    // 检查尺寸有效性
    if (dftsize.width <= 0 || dftsize.height <= 0) {
        std::cerr << "dfttemp: 无效的DFT尺寸" << std::endl;
        _dftTempl = cv::Mat();  // 返回空矩阵表示失败
        return;
    }

    // 创建DFT模板矩阵（初始化为0）
    cv::Mat dftTempl(dftsize.height, dftsize.width, maxDepth, cv::Scalar::all(0));

    // 定义操作区域
    cv::Mat src = templ;
    cv::Mat dst(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));  // 整个DFT区域
    cv::Mat dst1(dftTempl, cv::Rect(0, 0, templ.cols, templ.rows));       // 模板有效区域

    // 复制模板数据到DFT矩阵（需深度匹配）
    if (dst1.data != src.data) {  // 地址不同时才转换，避免冗余操作
        src.convertTo(dst1, dst1.depth());
    }

    // 填充右侧空白区域为0（若DFT宽度大于模板宽度）
    if (dst.cols > templ.cols) {
        cv::Mat part(dst,
            cv::Range(0, templ.rows),  // 行范围：0到模板高度
            cv::Range(templ.cols, dst.cols));  // 列范围：模板宽度到DFT宽度
        part = cv::Scalar::all(0);  // 填充0
    }

    // 执行DFT变换（第三个参数0表示默认选项，第四个参数为模板行数）
    cv::dft(dst, dst, 0, templ.rows);

    // 截取有效DFT区域并赋值给输出
    cv::Mat dftTempl1(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
    _dftTempl = dftTempl1.clone();  // 克隆确保数据独立
}

void NccMatch::crossCorr1(const cv::Mat& img, const cv::Mat& _dftTempl, cv::Mat& corr, const cv::Mat& _dftImg,
    cv::Size dftsize, int ctype, int maxDepth, double delta, int borderType)
{
    int cdepth = CV_MAT_DEPTH(ctype);

    cv::Size wholeSize = img.size();
    cv::Point roiofs(0, 0);
    cv::Mat img0 = img;
    if (!(borderType & cv::BORDER_ISOLATED))
    {
        img.locateROI(wholeSize, roiofs);
        img0.adjustROI(roiofs.y, wholeSize.height - img.rows - roiofs.y,
            roiofs.x, wholeSize.width - img.cols - roiofs.x);
    }

    borderType |= cv::BORDER_ISOLATED;
    cv::Mat dftImg(dftsize, maxDepth);
    cv::mulSpectrums(_dftImg, _dftTempl, dftImg, 0, true);
    cv::dft(dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, corr.rows);
    cv::Mat src1 = img0;
    src1 = dftImg(cv::Rect(0, 0, corr.cols, corr.rows));
    src1.convertTo(corr, cdepth, 1, delta);
}
bool NccMatch::bestTemplate(const std::vector<cv::Mat>& image, const int& method, int& result)
{
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    std::vector<std::pair<double, int>> Val;
    for (unsigned int i = 0; i < image.size(); i++)
    {
        cv::minMaxLoc(image.at(i), &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        Val.push_back(std::pair<double, int>(minVal, i));
    }
    std::sort(Val.begin(), Val.end());
    for (int i = 0; i < Val.size(); i++)
    {
        if (Val[i].first != 0)
        {
            result = Val[i].second;
            break;
        }
    }
    return true;
}
bool NccMatch::matchAngleTemplate(const cv::Mat& grayImage, std::vector<cv::Mat>& results) {
    
    if (fourierTemplates_.empty()) {
        std::cerr << "未生成模板，请先调用生成模板方法" << std::endl;
        return false;
    }

    // 初始化结果容器
    //results.resize(fourierTemplates_.size());
    int smalltemplate = 4;
    //这里压缩了两级
    cv::Mat small_image = grayImage.clone(); // 缩小4倍图像
    for (int i = 1; i < smalltemplate; i *= 2)
    {
        cv::pyrDown(small_image, small_image, cv::Size(small_image.cols / 2, small_image.rows / 2));
    }
    //--------4、获取目标缩小图像的边缘图像-----------//
    cv::Mat small_image_edge;
    cv::Canny(small_image, small_image_edge, 130, 200, 3, true);
    bool has_no_edge = (countNonZero(small_image_edge) == 0);
    if (has_no_edge)
    {
        // 图像预处理增强
         // 预处理：中值滤波去除椒盐噪声
        cv::Mat medianFilteredImage;
        cv::medianBlur(small_image, medianFilteredImage, 3);
        cv::Mat preprocessedImage;
        cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

        // 使用双边滤波减少噪声
        cv::Mat smoothedImage2;
        cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
        //cv::Canny(smoothedImage2, edges2, 200, 230, 3, true);
        cv::Canny(smoothedImage2, small_image_edge, 170, 230, 3, true);
    }

    // 对目标边缘所辖图像进行距离变换
    cv::Mat small_image_edge_fanse;
    cv::bitwise_not(small_image_edge, small_image_edge_fanse, cv::noArray());
    //--------5、距离变换-----------//
    cv::Mat small_image_edge_distance = cv::Mat::zeros(small_image.size(), CV_32FC1);
    cv::distanceTransform(small_image_edge_fanse, small_image_edge_distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    // --------利用小模板与小目标图像，进行带角度模板匹配---------- -//
    std::vector<cv::Mat> selected_small_templ_dft;
    double small_thearange[2] = { -5, 5 }; // 匹配角度范围
    double small_theastep = 1;  // 模板获取间隔值。
    for (int i = small_thearange[0]; i <= small_thearange[1]; i += small_theastep)
    {
        selected_small_templ_dft.push_back(fourierTemplates_.at(i - small_thearange[0]));
    }
    results.resize(selected_small_templ_dft.size());
    //std::vector<cv::Mat> small_match_result(selected_small_templ_dft.size());
    int small_match_method = cv::TM_CCORR;
    if (false == matchFttTemplate(small_image_edge_distance, selected_small_templ_dft, corrSize_, results, small_match_method))
    {
        return false;
    }
   

    // 检查尺寸匹配
    //if (imgDft.size() != fourierTemplates_[0].size()) {
    //    std::cerr << "模板尺寸与图像DFT尺寸不匹配" << std::endl;
    //    return false;
    //}

    // 对每个角度模板计算互相关
    /*for (size_t i = 0; i < fourierTemplates_.size(); ++i) {
        results[i].create(cv::Size(grayImage.cols - spatialTemplates_[i].first.cols + 1,
            grayImage.rows - spatialTemplates_[i].first.rows + 1), CV_32F);
        crossCorr1(grayImage, fourierTemplates_[i], results[i], imgDft);
    }*/

    return true;
}

bool NccMatch::matchFttTemplate(const cv::Mat& img, std::vector<cv::Mat>& tempdft, cv::Size corrsize, std::vector<cv::Mat>& result, const int& method)
{

    cv::Mat dft_img;
    cv::Size dftsize;
    int maxDepth = 0;

    cv::Mat oneresult;
    dftimg(img, oneresult, dft_img,
        corrsize, dftsize,
        CV_32F, maxDepth, 0, 0);

    if (dftsize.width != tempdft.at(0).cols || dftsize.height != tempdft.at(0).rows)
        return false;

    for (int i = 0; i < tempdft.size(); i++)
    {
        result.at(i).create(cv::Size(oneresult.cols, oneresult.rows), CV_32F);
        crossCorr1(img, tempdft.at(i), result.at(i), dft_img, dftsize, CV_32F, maxDepth, 0, 0);
    }

    return true;
}
bool NccMatch::matchNCCTemplate(const cv::Mat& img, const std::vector<cv::Mat>& templ, std::vector<cv::Mat>& result, const int& method)
{
    for (int i = 0; i < templ.size(); i++)
    {
        cv::Mat templFloat;  // 用于存储转换后的float类型模板
        templ[i].convertTo(templFloat, CV_32F, 1.0 / 255.0);  // 正确：输出到新矩阵
        cv::matchTemplate(img, templFloat, result.at(i), method);
    }
    return true;
}
bool NccMatch::findBestMatch(const std::vector<cv::Mat>& results, cv::Point2f& bestLoc,
    float& bestAngle, double& bestScore) {
    bestScore = -1.0;
    bestLoc = cv::Point2f(-1, -1);
    bestAngle = 0.0f;

    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].empty()) continue;

        // 找到当前角度的最佳匹配
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(results[i], &minVal, &maxVal, &minLoc, &maxLoc);

         //更新全局最佳匹配
        if (maxVal > bestScore) {
            bestScore = maxVal;
            // 计算中心坐标（加上模板半尺寸）
            bestLoc.x = maxLoc.x + spatialTemplates_[i].first.cols / 2.0f;
            bestLoc.y = maxLoc.y + spatialTemplates_[i].first.rows / 2.0f;
            bestAngle = spatialTemplates_[i].second;
        }
    }

    return bestScore > 0;  // 得分大于0视为有效匹配
}

bool  NccMatch::matchLocation(cv::Mat& image, const int& method, std::vector<std::pair<double, cv::Point2f>>& location, int x, int y, double& val)
{
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    minMaxLoc(image, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
    location.push_back(std::pair<double, cv::Point2f>(minVal, (cv::Point2f)minLoc + cv::Point2f((float)(x/*-1*/) / 2, (float)(y/*-1*/) / 2)));
    image.at<float>(minLoc) = maxVal;
    val = minVal;
    return true;
}
// 实现从XML加载边缘点并生成模板的方法
bool NccMatch::loadEdgePointsAndGenerateTemplates(const std::string& edgeXmlPath, const cv::Mat& Image) {

    // 2. 从XML加载边缘点
    std::vector<cv::Point> edgePoints;
    std::vector<cv::Point2f> subedgePoints;
    // 打开XML文件
    cv::FileStorage fs(edgeXmlPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    // 读取边缘点
    cv::FileNode edgeNode = fs["edgePoints"];
    if (edgeNode.type() != cv::FileNode::SEQ) {
        std::cerr << "edgePoints不是序列类型" << std::endl;
        return false;
    }

    cv::FileNodeIterator it = edgeNode.begin();
    for (; it != edgeNode.end(); ++it) {
        cv::Point pt;
        (*it)["x"] >> pt.x;
        (*it)["y"] >> pt.y;
        edgePoints.push_back(pt);
    }

    // 读取亚像素点
    cv::FileNode subEdgeNode = fs["subedgePoints"];
    if (subEdgeNode.type() != cv::FileNode::SEQ) {
        return false;
    }

    it = subEdgeNode.begin();
    for (; it != subEdgeNode.end(); ++it) {
        cv::Point2f pt;
        (*it)["x"] >> pt.x;
        (*it)["y"] >> pt.y;
        subedgePoints.push_back(pt);
    }

    fs.release();

    // 4. 用加载的边缘点生成模板（复用原有generateTemplates方法）
    return generateTemplates1(edgePoints, subedgePoints, Image);
}
// 实现完整匹配流程函数（含边缘点加载）
bool NccMatch::runFullMatchingFromPath(
    const cv::Mat& grayImage,
    const std::string& edgeXmlPath,
    cv::Point2f& bestLoc,
    float& bestAngle,
    double& bestScore
) {
    try {
        // 1. 初始化输出参数
        bestLoc = cv::Point2f(-1, -1);
        bestAngle = 0.0f;
        bestScore = -1.0;
        // 3. 读取待匹配图像并转换为灰度图
       // cv::Mat testImg = cv::imread(testImagePath);
        cv::Mat testImg = grayImage.clone();
       
        cv::Mat testGray;
        if (testImg.channels() == 3) {
            cv::cvtColor(testImg, testGray, cv::COLOR_BGR2GRAY);
        }
        else if (testImg.channels() == 1) {
            testGray = testImg.clone();
        }
        else {
            return false;
        }
        // 2. 从XML加载边缘点并生成模板（核心新增步骤）
        if (!loadEdgePointsAndGenerateTemplates(edgeXmlPath, testGray)) {
            return false;
        }
        // 4. 执行匹配和结果解析（复用原有逻辑）
        std::vector<cv::Mat> matchResults;
        if (!matchAngleTemplate(testGray, matchResults)) {
            return false;
        }
        int small_best_template_index;
        int small_match_method = cv::TM_CCORR;
        bestTemplate(matchResults, small_match_method, small_best_template_index);
        std::vector<cv::Mat> selected_templ_dft;
        double small_thearange[2] = { -1, 1 }; // 匹配角度范围
        int index1 = 0;
        for (int i = 0; i < 3; i++)
        {
            if (small_best_template_index > 0)
            {
                index1 = int(small_best_template_index + i - small_thearange[1]);
            }
            else
            {
                index1 = int(small_best_template_index + i);
            }
            index1 = index1 < 11 ? index1 : 10;
            const auto& tempGroup = template_pyrdown_[index1];
            cv::Mat rotate_template_image = tempGroup.first[0];
            selected_templ_dft.push_back(rotate_template_image);
        }
        std::vector<cv::Mat> match_result(selected_templ_dft.size());
        //进行canny边缘检测
        cv::Mat image_edge;
        cv::Canny(testGray, image_edge, 140, 200, 3, true);
        // 判断图像是否全黑（非零像素数为0，Canny输出的边缘图中255为边缘点）
        bool isImageBlack = (cv::countNonZero(image_edge) == 0);
        if (isImageBlack)
        {
            // 预处理：中值滤波去除椒盐噪声
            cv::Mat medianFilteredImage;
            cv::medianBlur(testGray, medianFilteredImage, 3);
            cv::Mat preprocessedImage;
            cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

            // 使用双边滤波减少噪声
            cv::Mat smoothedImage2, edges2;
            cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
            //cv::Canny(smoothedImage2, edges2, 200, 230, 3, true);
            cv::Canny(smoothedImage2, image_edge, 170, 230, 3, true);
        }
       
        //cv::Canny(testGray, image_edge, 140, 200, 3, true);
        // 对目标边缘图像进行距离变换
        cv::Mat image_edge_fanse;
        cv::bitwise_not(image_edge, image_edge_fanse, cv::noArray());


        //-------------10、这里进行一步距离变换，并对距离求取指数值变换------------//
        cv::Mat image_edge_distance = cv::Mat::zeros(testGray.size(), CV_32FC1);
        cv::distanceTransform(image_edge_fanse, image_edge_distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        exp(-0.2 * image_edge_distance, image_edge_distance);
        image_edge_distance = 1 - image_edge_distance;
        int match_method = cv::TM_CCORR;
        matchNCCTemplate(image_edge_distance, selected_templ_dft, match_result, match_method);
        int best_template_index;
        bestTemplate(match_result, match_method, best_template_index);
        cv::Mat rotate_template_image = selected_templ_dft.at(best_template_index);
        bestAngle = small_best_template_index + best_template_index-6 ;
        std::vector<std::pair<double, cv::Point2f>> precise_location;
        double precise_val = 0.0;
        matchLocation(match_result.at(best_template_index), match_method, precise_location, rotate_template_image.cols, rotate_template_image.rows, precise_val);
        
        //位置计算
        bestLoc.x = precise_location[0].second.x;
        bestLoc.y = precise_location[0].second.y;
        //亚像素匹配
        vector<Point2f> rotated_points= rotatedSubedgePoints_[small_best_template_index].first;
        sm::imgproc::Scene_edge scene;
        vector<sm::imgproc::Vec2f> pcd_buffer, normal_buffer;
        scene.init_Scene_edge(testGray, pcd_buffer, normal_buffer);
        std::vector<sm::imgproc::Vec2f> model_pcd(rotated_points.size());
        for (int i = 0; i < rotated_points.size(); i++)
        {
            auto& feat = rotated_points;
            model_pcd[i] = { float(rotated_points[i].x + (bestLoc.x - rotate_template_image.cols / 2)), float(rotated_points[i].y + (bestLoc.y - rotate_template_image.rows / 2)) };
        }
        sm::imgproc::icp::RegistrationResult Reresult = sm::imgproc::icp::ICP2D_Point2Plane(model_pcd, scene);
        float x_refine = Reresult.transformation_[0][0] * (bestLoc.x - rotate_template_image.cols / 2) + Reresult.transformation_[0][1] * (bestLoc.y - rotate_template_image.rows / 2) + Reresult.transformation_[0][2];
        float y_refine = Reresult.transformation_[1][0] * (bestLoc.x - rotate_template_image.cols / 2) + Reresult.transformation_[1][1] * (bestLoc.y - rotate_template_image.rows / 2) + Reresult.transformation_[1][2];
        float cos_r = Reresult.transformation_[0][0];
        float sin_r = Reresult.transformation_[1][0];
        float r_refine = std::atan2(sin_r, cos_r); // r 是旋转角，单位是弧度
        float r_deg = r_refine * 180.0f / CV_PI; // 弧度转角度
        //bestLoc.x += spatialTemplates_[i].first.cols / 2.0f;
        //bestLoc.y += spatialTemplates_[i].first.rows / 2.0f;
;

        //绘图
        // 从XML加载边缘点（单独读取用于绘制）
       
#ifdef DRAW_MATCH_EDGE
        std::vector<cv::Point> templateEdgePoints;

// 遍历图像的每个像素
        for (int y = 0; y < rotate_template_image.rows; ++y)
        {
            // 获取当前行的指针
            const uchar* rowPtr = rotate_template_image.ptr<uchar>(y);

            for (int x = 0; x < rotate_template_image.cols; ++x)
            {
                // 边缘点通常值为255（白色）
                if (rowPtr[x] >= 60)
                {
                    templateEdgePoints.push_back(cv::Point(x, y));
                }
            }
        }
            // 3. 绘制边缘点（蓝色小点）
        for (const auto& pt : templateEdgePoints) {
            // 计算边缘点在匹配位置的对应坐标（考虑旋转和平移）
             // 转换为相对于中心点的坐标
            //cv::Mat templ_imag= match_result.at(best_template_index);
            float dx = pt.x + (bestLoc.x- rotate_template_image.cols/2);
            float dy = pt.y +(bestLoc.y- rotate_template_image.rows/2);

            // 旋转变换公式
            float cosA = cos(0);
            float sinA = sin(0);
            float x = dx * cosA - dy * sinA + bestLoc.x;
            float y = dx * sinA + dy * cosA + bestLoc.y;
            cv::Point2f transformedPt;
            transformedPt.x = dx;
            transformedPt.y = dy;
            //cv::circle(testImg, transformedPt, 1, cv::Scalar(255, 0, 0), -1);
            cv::line(testImg,
                cv::Point(dx, dy),
                cv::Point(dx, dy),
                cv::Scalar(255, 0, 0), 1);
        }
#endif
        bestLoc.x = x_refine+rotate_template_image.cols / 2;
        bestLoc.y = y_refine+ rotate_template_image.rows / 2;
        int sumIndex = small_best_template_index + best_template_index;

        // 特殊情况处理：索引和为0时，按要求赋值
        if (sumIndex == 0) {
            bestAngle = -5 - r_deg;
        }
        else {
            // 原逻辑：正常情况下的角度计算
            bestAngle = sumIndex - 6 - r_deg;
        }
        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

// -------------------------- 从已加载的灰度图执行匹配 --------------------------
bool NccMatch::runFullMatchingFromGrayMat(
    const cv::Mat& testGray,
    cv::Point2f& bestLoc,
    float& bestAngle,
    double& bestScore
) {
    try {
        // 1. 初始化输出参数
        bestLoc = cv::Point2f(-1, -1);
        bestAngle = 0.0f;
        bestScore = -1.0;

        // 2. 校验输入灰度图有效性
        if (testGray.empty()) {
            std::cerr << "[错误] 输入灰度图为空（未加载或加载失败）" << std::endl;
            return false;
        }
        if (testGray.type() != CV_8UC1) {
            std::cerr << "[错误] 输入图像非单通道灰度图（需CV_8UC1，当前类型：" << testGray.type() << "）" << std::endl;
            return false;
        }

        // 3. 校验模板是否已生成（避免未生成模板就匹配）
        if (spatialTemplates_.empty() || fourierTemplates_.empty()) {
            std::cerr << "[错误] 未生成匹配模板，请先调用 generateTemplateAndSaveEdgePoints 生成模板" << std::endl;
            return false;
        }

        // 4. 执行核心匹配（调用原有 matchAngleTemplate 方法）
        std::vector<cv::Mat> matchResults;
        bool matchSuccess = matchAngleTemplate(testGray, matchResults);
        if (!matchSuccess) {
            std::cerr << "[错误] 傅里叶域匹配执行失败（可能是模板尺寸与图像尺寸不匹配）" << std::endl;
            return false;
        }

        // 5. 解析最佳匹配结果（调用原有 findBestMatch 方法）
        bool findSuccess = findBestMatch(matchResults, bestLoc, bestAngle, bestScore);
        if (!findSuccess) {
            std::cerr << "[提示] 未找到有效匹配（所有角度的匹配得分均低于阈值）" << std::endl;
            return false;
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[异常] 匹配流程执行失败：" << e.what() << std::endl;
        return false;
    }
}

//temp
// 使用OpenCL加速的Canny边缘检测
double NccMatch::cannyWithOpenCL(const cv::Mat& input, cv::Mat& output, double threshold1, double threshold2) {
    // 启用OpenCL
    cv::ocl::setUseOpenCL(true);

    if (!cv::ocl::haveOpenCL()) {
        std::cerr << "OpenCL is not available on this system!" << std::endl;
        return -1;
    }

    // 转换为UMat以利用OpenCL加速
    cv::UMat src_umat, dst_umat;
    input.copyTo(src_umat);

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 执行Canny边缘检测（会自动使用OpenCL加速）
    cv::Canny(src_umat, dst_umat, threshold1, threshold2);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 转换回Mat
    dst_umat.copyTo(output);

    // 计算并返回耗时（毫秒）
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// 不使用OpenCL的Canny边缘检测
double NccMatch::cannyWithoutOpenCL(const cv::Mat& input, cv::Mat& output, double threshold1, double threshold2) {
    // 禁用OpenCL
    cv::ocl::setUseOpenCL(false);

    cv::Mat src_mat = input.clone();
    cv::Mat dst_mat;

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 执行Canny边缘检测（纯CPU计算）
    cv::Canny(src_mat, dst_mat, threshold1, threshold2);

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 保存结果
    output = dst_mat;

    // 计算并返回耗时（毫秒）
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// 使用默认方法（可能启用IPP）进行图像处理
double NccMatch::processWithPossibleIPP(const cv::Mat& input, cv::Mat& output) {
    cv::Mat src = input.clone();
    cv::Mat dst;

    auto start = std::chrono::high_resolution_clock::now();

    // 执行一系列图像处理操作
    cv::GaussianBlur(src, dst, cv::Size(5, 5), 0);
    cv::Canny(dst, dst, 50, 150);
    cv::resize(dst, dst, cv::Size(), 0.5, 0.5);

    auto end = std::chrono::high_resolution_clock::now();

    output = dst;
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

// 尝试禁用IPP进行对比（注：OpenCV没有直接禁用IPP的API，这里通过修改配置尝试）
double NccMatch::processWithoutIPP(const cv::Mat& input, cv::Mat& output) {
    cv::Mat src = input.clone();
    cv::Mat dst;

    // 尝试禁用IPP优化（不保证在所有版本有效）
    cv::setUseOptimized(false);

    auto start = std::chrono::high_resolution_clock::now();

    // 执行相同的图像处理操作
    cv::GaussianBlur(src, dst, cv::Size(5, 5), 0);
    cv::Canny(dst, dst, 50, 150);
    cv::resize(dst, dst, cv::Size(), 0.5, 0.5);

    auto end = std::chrono::high_resolution_clock::now();

    // 恢复默认优化设置
    cv::setUseOptimized(true);

    output = dst;
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

//从边缘图像中提取非杂点边缘点（3x3邻域过滤）
void NccMatch::extractEdgePointsWithNoiseFilter(const cv::Mat& edgeImage, std::vector<cv::Point>& edgePoints,const int& minArea) {
    edgePoints.clear(); // 确保容器为空
    // 连通域分析：过滤面积过小的杂点（基于开运算后的图像）
    cv::Mat labels, stats, centroids;
    
    int numLabels = cv::connectedComponentsWithStats(edgeImage, labels, stats, centroids, 8);

    // 过滤小连通域（设定面积阈值，根据实际场景调整）
    cv::Mat filteredEdges = cv::Mat::zeros(edgeImage.size(), CV_8UC1);

    for (int i = 1; i < numLabels; ++i) { // i=0是背景，从1开始遍历前景连通域
        int area = stats.at<int>(i, cv::CC_STAT_AREA); // 获取第i个连通域的面积
        if (area >= minArea) { // 保留面积大于等于阈值的连通域
            // 将该连通域的所有像素标记为边缘（255）
            filteredEdges.setTo(255, labels == i);
        }
    }
    for (int y = 0; y < filteredEdges.rows; ++y) {
        const uchar* rowPtr = filteredEdges.ptr<uchar>(y);
        for (int x = 0; x < filteredEdges.cols; ++x) {
            if (rowPtr[x] == 255) { // 当前点是边缘点
                bool isNoise = true;
                // 3x3邻域检查（排除自身）
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue; // 跳过自身

                        int nx = x + dx;
                        int ny = y + dy;
                        // 检查邻点是否在图像范围内且为边缘点
                        if (nx >= 0 && nx < filteredEdges.cols && ny >= 0 && ny < filteredEdges.rows) {
                            if (filteredEdges.at<uchar>(ny, nx) == 255) {
                                isNoise = false;
                                break; // 找到邻域边缘点，跳出dx循环
                            }
                        }
                    }
                    if (!isNoise) break; // 跳出dy循环
                }
                if (!isNoise) {
                    edgePoints.push_back(cv::Point(x, y));
                }
            }
        }
    }
}

////测试水平垂直线段位置
bool NccMatch::Region_test(
    const cv::Mat& grayImage,
    const std::string& edgeXmlPath,
    cv::Point2f& bestLoc,
    float& bestAngle,
    double& bestScore
) {
    try {
        // 1. 初始化输出参数
        bestLoc = cv::Point2f(-1, -1);
        bestAngle = 0.0f;
        bestScore = -1.0;
        // 3. 读取待匹配图像并转换为灰度图
       // cv::Mat testImg = cv::imread(testImagePath);
        cv::Mat testImg = grayImage.clone();

        cv::Mat testGray;
        if (testImg.channels() == 3) {
            cv::cvtColor(testImg, testGray, cv::COLOR_BGR2GRAY);
        }
        else if (testImg.channels() == 1) {
            testGray = testImg.clone();
        }
        else {
            return false;
        }
        //进行canny边缘检测
        cv::Mat image_edge;
        cv::Canny(testGray, image_edge, 140, 200, 3, true);
        // 判断图像是否全黑（非零像素数为0，Canny输出的边缘图中255为边缘点）
        bool isImageBlack = (cv::countNonZero(image_edge) == 0);
        if (isImageBlack)
        {
            // 预处理：中值滤波去除椒盐噪声
            cv::Mat medianFilteredImage;
            cv::medianBlur(testGray, medianFilteredImage, 3);
            cv::Mat preprocessedImage;
            cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

            // 使用双边滤波减少噪声
            cv::Mat smoothedImage2, edges2;
            cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
            //cv::Canny(smoothedImage2, edges2, 200, 230, 3, true);
            cv::Canny(smoothedImage2, image_edge, 170, 230, 3, true);
        }
        
                 // 5. 检测外层轮廓
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(image_edge.clone(), contours, hierarchy,
            cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 6. 创建轮廓图像
        cv::Mat contourImage = cv::Mat::zeros(image_edge.size(), CV_8UC1);
        cv::drawContours(contourImage, contours, -1, cv::Scalar(255), 1);

        // 7. 计算所有轮廓的最小外接矩形
        if (contours.empty()) {
            return false;
        }

        // 将所有轮廓点合并
        std::vector<cv::Point> allPoints;
        for (const auto& contour : contours) {
            allPoints.insert(allPoints.end(), contour.begin(), contour.end());
        }
        std::vector<cv::Point2f> subedgePoints;
        SubPixelByZernike(grayImage, allPoints, subedgePoints);

        // 计算最小外接矩形
        cv::RotatedRect minRect = cv::minAreaRect(allPoints);
        cv::Rect boundingRect = minRect.boundingRect();

        // 确保矩形在图像范围内
        boundingRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);

        if (boundingRect.width <= 0 || boundingRect.height <= 0) {
            return false;
        }

        // 8. 存储4条边的坐标（映射到原始图像）
        std::vector<float> edgeCoordinates(4, -1); // 0:上边y, 1:下边y, 2:左边x, 3:右边x

        // 外接矩形参数
        int rectX = boundingRect.x;
        int rectY = boundingRect.y;
        int rectWidth = boundingRect.width;
        int rectHeight = boundingRect.height;

        // 每条边取中间40%的区域
        float edgeRatio = 0.4f;

        // 存储每个区域的边缘点
        std::vector<std::vector<cv::Point>> regionPoints(4);

        // 9. 上边缘区域（中间40%）
        int topRegionWidth = static_cast<int>(rectWidth * edgeRatio);
        int topStartX = rectX + (rectWidth - topRegionWidth) / 2;
        int topEndX = topStartX + topRegionWidth;

        // 上边区域的高度取一小段（比如10像素）
        int topHeight = 10;
        cv::Rect topRect(topStartX, rectY- (topHeight/2), topRegionWidth, topHeight);
        topRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);
        cv::Mat topROI;
        if (topRect.width > 0 && topRect.height > 0) {
            topROI = contourImage(topRect);

            // 提取上边缘点
            std::vector<cv::Point> topPoints;
            cv::findNonZero(topROI, topPoints);

            if (!topPoints.empty()) {
                // 将点坐标映射回原始图像
                for (auto& pt : topPoints) {
                    pt.x += topRect.x;
                    pt.y += topRect.y;
                    regionPoints[0].push_back(pt);
                }
            }
        }

        // 10. 下边缘区域（中间40%）
        int bottomStartX = topStartX;  // 与上边对称
        int bottomEndX = topEndX;
        int bottomY = rectY + rectHeight - topHeight;

        cv::Rect bottomRect(bottomStartX, bottomY+ (topHeight / 2), topRegionWidth, topHeight);
        bottomRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);
        cv::Mat bottomROI;
        if (bottomRect.width > 0 && bottomRect.height > 0) {
             bottomROI = contourImage(bottomRect);

            // 提取下边缘点
            std::vector<cv::Point> bottomPoints;
            cv::findNonZero(bottomROI, bottomPoints);

            if (!bottomPoints.empty()) {
                // 将点坐标映射回原始图像
                for (auto& pt : bottomPoints) {
                    pt.x += bottomRect.x;
                    pt.y += bottomRect.y;
                    regionPoints[1].push_back(pt);
                }
            }
        }

        // 11. 左边缘区域（中间40%）
        int leftRegionHeight = static_cast<int>(rectHeight * edgeRatio);
        int leftStartY = rectY + (rectHeight - leftRegionHeight) / 2;
        int leftEndY = leftStartY + leftRegionHeight;

        // 左边区域的宽度取一小段（10像素）
        int leftWidth = 10;
        cv::Rect leftRect(rectX-(leftWidth/2), leftStartY, leftWidth, leftRegionHeight);
        leftRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);
        cv::Mat leftROI;
        if (leftRect.width > 0 && leftRect.height > 0) {
            leftROI = contourImage(leftRect);

            // 提取左边缘点
            std::vector<cv::Point> leftPoints;
            cv::findNonZero(leftROI, leftPoints);

            if (!leftPoints.empty()) {
                // 将点坐标映射回原始图像
                for (auto& pt : leftPoints) {
                    pt.x += leftRect.x;
                    pt.y += leftRect.y;
                    regionPoints[2].push_back(pt);
                }
            }
        }

        // 12. 右边缘区域（中间40%）
        int rightStartY = leftStartY;  // 与左边对称
        int rightEndY = leftEndY;
        int rightX = rectX + rectWidth - leftWidth;

        cv::Rect rightRect(rightX+ (leftWidth / 2), rightStartY, leftWidth, leftRegionHeight);
        rightRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);
        cv::Mat rightROI;
        if (rightRect.width > 0 && rightRect.height > 0) {
            rightROI = contourImage(rightRect);

            // 提取右边缘点
            std::vector<cv::Point> rightPoints;
            cv::findNonZero(rightROI, rightPoints);

            if (!rightPoints.empty()) {
                // 将点坐标映射回原始图像
                for (auto& pt : rightPoints) {
                    pt.x += rightRect.x;
                    pt.y += rightRect.y;
                    regionPoints[3].push_back(pt);
                }
            }
        }

        // 13. 对每个区域的边缘点进行更精确的直线拟合
        for (int i = 0; i < 4; ++i) {
            if (regionPoints[i].size() < 2) {
                continue;  // 点数太少，无法拟合
            }

            // 使用最小二乘法拟合直线
            cv::Vec4f line;
            cv::fitLine(regionPoints[i], line, cv::DIST_L2, 0, 0.01, 0.01);

            // line[0], line[1] 是方向向量
            // line[2], line[3] 是直线上的一点

            float vx = line[0];
            float vy = line[1];
            float x0 = line[2];
            float y0 = line[3];

            if (i == 0 || i == 1) { // 上边或下边：需要水平线，y坐标为常数
                // 计算拟合直线在区域中心x处的y值
                float centerX = (i == 0) ? (topStartX + topRegionWidth / 2.0f) : (bottomStartX + topRegionWidth / 2.0f);

                // 直线方程: (x - x0)/vx = (y - y0)/vy
                // 所以 y = y0 + (x - x0) * vy / vx

                // vx可能为0（垂直线），
                if (std::abs(vx) > 1e-5) {
                    float y = y0 + (centerX - x0) * vy / vx;
                    edgeCoordinates[i] = y;
                }
                else {
                    // 如果是垂直线，取所有点的平均y值
                    float sumY = 0;
                    for (const auto& pt : regionPoints[i]) {
                        sumY += pt.y;
                    }
                    edgeCoordinates[i] = sumY / regionPoints[i].size();
                }
            }
            else { // 左边或右边：需要垂直线，x坐标为常数
             // 计算拟合直线在区域中心y处的x值
                float centerY = (i == 2) ? (leftStartY + leftRegionHeight / 2.0f) : (rightStartY + leftRegionHeight / 2.0f);
                if (std::abs(vy) > 1e-5) {
                    float x = x0 + (centerY - y0) * vx / vy;
                    edgeCoordinates[i] = x;
                }
                else {
                    // 如果是水平线，取所有点的平均x值
                    float sumX = 0;
                    for (const auto& pt : regionPoints[i]) {
                        sumX += pt.x;
                    }
                    edgeCoordinates[i] = sumX / regionPoints[i].size();
                }
            }

            // 如果拟合效果不好，使用原始的平均值方法
            if (edgeCoordinates[i] < 0) {
                if (i == 0 || i == 1) {
                    float sumY = 0;
                    for (const auto& pt : regionPoints[i]) {
                        sumY += pt.y;
                    }
                    edgeCoordinates[i] = sumY / regionPoints[i].size();
                }
                else {
                    float sumX = 0;
                    for (const auto& pt : regionPoints[i]) {
                        sumX += pt.x;
                    }
                    edgeCoordinates[i] = sumX / regionPoints[i].size();
                }
            }
        }

        // 14. 计算上下距离和左右距离
        float verticalDistance = -1.0f;  // 上下距离
        float horizontalDistance = -1.0f; // 左右距离

        if (edgeCoordinates[0] >= 0 && edgeCoordinates[1] >= 0) {
            verticalDistance = std::abs(edgeCoordinates[1] - edgeCoordinates[0]);
        }

        if (edgeCoordinates[2] >= 0 && edgeCoordinates[3] >= 0) {
            horizontalDistance = std::abs(edgeCoordinates[3] - edgeCoordinates[2]);
        }
        bestLoc.x = edgeCoordinates[2]+(horizontalDistance/2);
        bestLoc.y = edgeCoordinates[0]+(verticalDistance/2);
        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

bool NccMatch::Region_test_subpix(
    const cv::Mat& grayImage,
    const std::string& edgeXmlPath,
    cv::Point2f& bestLoc,
    float& bestAngle,
    double& bestScore
) {
    try {
        // 1. 初始化输出参数
        bestLoc = cv::Point2f(-1, -1);
        bestAngle = 0.0f;
        bestScore = -1.0;
        // 3. 读取待匹配图像并转换为灰度图
       // cv::Mat testImg = cv::imread(testImagePath);
        cv::Mat testImg = grayImage.clone();

        cv::Mat testGray;
        if (testImg.channels() == 3) {
            cv::cvtColor(testImg, testGray, cv::COLOR_BGR2GRAY);
        }
        else if (testImg.channels() == 1) {
            testGray = testImg.clone();
        }
        else {
            return false;
        }
        //进行canny边缘检测
        cv::Mat image_edge;
        cv::Canny(testGray, image_edge, 140, 200, 3, true);
        // 判断图像是否全黑（非零像素数为0，Canny输出的边缘图中255为边缘点）
        bool isImageBlack = (cv::countNonZero(image_edge) == 0);
        if (isImageBlack)
        {
            // 预处理：中值滤波去除椒盐噪声
            cv::Mat medianFilteredImage;
            cv::medianBlur(testGray, medianFilteredImage, 3);
            cv::Mat preprocessedImage;
            cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

            // 使用双边滤波减少噪声
            cv::Mat smoothedImage2, edges2;
            cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
            //cv::Canny(smoothedImage2, edges2, 200, 230, 3, true);
            cv::Canny(smoothedImage2, image_edge, 170, 230, 3, true);
        }

        // 5. 检测外层轮廓
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(image_edge.clone(), contours, hierarchy,
            cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 6. 创建轮廓图像
        cv::Mat contourImage = cv::Mat::zeros(image_edge.size(), CV_8UC1);
        cv::drawContours(contourImage, contours, -1, cv::Scalar(255), 1);

        // 7. 计算所有轮廓的最小外接矩形
        if (contours.empty()) {
            return false;
        }

        // 将所有轮廓点合并
        std::vector<cv::Point> allPoints;
        cv::findNonZero(image_edge, allPoints);
        std::vector<cv::Point2f> subedgePoints;
        SubPixelByZernike(grayImage, allPoints, subedgePoints);

        // 计算最小外接矩形（使用整数点）
        cv::RotatedRect minRect = cv::minAreaRect(allPoints);
        cv::Rect boundingRect = minRect.boundingRect();

        // 确保矩形在图像范围内
        boundingRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);

        if (boundingRect.width <= 0 || boundingRect.height <= 0) {
            return false;
        }

        // 8. 存储4条边的坐标（映射到原始图像）
        std::vector<float> edgeCoordinates(4, -1); // 0:上边y, 1:下边y, 2:左边x, 3:右边x

        // 外接矩形参数
        int rectX = boundingRect.x;
        int rectY = boundingRect.y;
        int rectWidth = boundingRect.width;
        int rectHeight = boundingRect.height;

        // 每条边取中间40%的区域
        float edgeRatio = 0.4f;

        // 存储每个区域的亚像素边缘点
        std::vector<std::vector<cv::Point2f>> regionSubPoints(4);

        // 9. 上边缘区域（中间40%）
        int topRegionWidth = static_cast<int>(rectWidth * edgeRatio);
        int topStartX = rectX + (rectWidth - topRegionWidth) / 2;
        int topEndX = topStartX + topRegionWidth;

        // 上边区域的高度取一小段（比如10像素）
        int topHeight = 10;
        cv::Rect topRect(topStartX, rectY - (topHeight / 2), topRegionWidth, topHeight);
        topRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);

        // 10. 下边缘区域（中间40%）
        int bottomStartX = topStartX;  // 与上边对称
        int bottomEndX = topEndX;
        int bottomY = rectY + rectHeight - topHeight;

        cv::Rect bottomRect(bottomStartX, bottomY + (topHeight / 2), topRegionWidth, topHeight);
        bottomRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);

        // 11. 左边缘区域（中间40%）
        int leftRegionHeight = static_cast<int>(rectHeight * edgeRatio);
        int leftStartY = rectY + (rectHeight - leftRegionHeight) / 2;
        int leftEndY = leftStartY + leftRegionHeight;

        // 左边区域的宽度取一小段（10像素）
        int leftWidth = 10;
        cv::Rect leftRect(rectX - (leftWidth / 2), leftStartY, leftWidth, leftRegionHeight);
        leftRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);

        // 12. 右边缘区域（中间40%）
        int rightStartY = leftStartY;  // 与左边对称
        int rightEndY = leftEndY;
        int rightX = rectX + rectWidth - leftWidth;

        cv::Rect rightRect(rightX + (leftWidth / 2), rightStartY, leftWidth, leftRegionHeight);
        rightRect &= cv::Rect(0, 0, contourImage.cols, contourImage.rows);

        // 将矩形转换为浮点版本用于点判断
        cv::Rect2f topRectF(topRect);
        cv::Rect2f bottomRectF(bottomRect);
        cv::Rect2f leftRectF(leftRect);
        cv::Rect2f rightRectF(rightRect);

        // 13. 从亚像素点中筛选属于各个区域的点
        for (const auto& pt : subedgePoints) {
            if (topRectF.contains(pt)) {
                regionSubPoints[0].push_back(pt);
            }
            else if (bottomRectF.contains(pt)) {
                regionSubPoints[1].push_back(pt);
            }
            else if (leftRectF.contains(pt)) {
                regionSubPoints[2].push_back(pt);
            }
            else if (rightRectF.contains(pt)) {
                regionSubPoints[3].push_back(pt);
            }
        }

        // 14. 对每个区域的亚像素边缘点进行直线拟合
        for (int i = 0; i < 4; ++i) {
            if (regionSubPoints[i].size() < 2) {
                continue;  // 点数太少，无法拟合
            }

            // 使用最小二乘法拟合直线
            cv::Vec4f line;

            // 将Point2f转换为Point用于fitLine（如果需要的话）
            // 或者直接使用Point2f，但需要确保fitLine支持
            // 这里我们使用Point2f转换为vector<Point2f>
            std::vector<cv::Point2f> pointsForFit = regionSubPoints[i];
            cv::fitLine(pointsForFit, line, cv::DIST_L2, 0, 0.01, 0.01);

            // line[0], line[1] 是方向向量
            // line[2], line[3] 是直线上的一点
            float vx = line[0];
            float vy = line[1];
            float x0 = line[2];
            float y0 = line[3];

            if (i == 0 || i == 1) { // 上边或下边：需要水平线，y坐标为常数
                // 计算拟合直线在区域中心x处的y值
                float centerX = (i == 0) ? (topRectF.x + topRectF.width / 2.0f) :
                    (bottomRectF.x + bottomRectF.width / 2.0f);

                // 直线方程: (x - x0)/vx = (y - y0)/vy
                // 所以 y = y0 + (x - x0) * vy / vx
                if (std::abs(vx) > 1e-5) {
                    float y = y0 + (centerX - x0) * vy / vx;
                    edgeCoordinates[i] = y;
                }
                else {
                    // 如果是垂直线，取所有点的平均y值
                    float sumY = 0;
                    for (const auto& pt : regionSubPoints[i]) {
                        sumY += pt.y;
                    }
                    edgeCoordinates[i] = sumY / regionSubPoints[i].size();
                }
            }
            else { // 左边或右边：需要垂直线，x坐标为常数
                // 计算拟合直线在区域中心y处的x值
                float centerY = (i == 2) ? (leftRectF.y + leftRectF.height / 2.0f) :
                    (rightRectF.y + rightRectF.height / 2.0f);
                if (std::abs(vy) > 1e-5) {
                    float x = x0 + (centerY - y0) * vx / vy;
                    edgeCoordinates[i] = x;
                }
                else {
                    // 如果是水平线，取所有点的平均x值
                    float sumX = 0;
                    for (const auto& pt : regionSubPoints[i]) {
                        sumX += pt.x;
                    }
                    edgeCoordinates[i] = sumX / regionSubPoints[i].size();
                }
            }

            // 如果拟合效果不好，使用原始的平均值方法
            if (edgeCoordinates[i] < 0) {
                if (i == 0 || i == 1) {
                    float sumY = 0;
                    for (const auto& pt : regionSubPoints[i]) {
                        sumY += pt.y;
                    }
                    edgeCoordinates[i] = sumY / regionSubPoints[i].size();
                }
                else {
                    float sumX = 0;
                    for (const auto& pt : regionSubPoints[i]) {
                        sumX += pt.x;
                    }
                    edgeCoordinates[i] = sumX / regionSubPoints[i].size();
                }
            }
        }

        // 15. 计算上下距离和左右距离
        float verticalDistance = -1.0f;  // 上下距离
        float horizontalDistance = -1.0f; // 左右距离

        if (edgeCoordinates[0] >= 0 && edgeCoordinates[1] >= 0) {
            verticalDistance = std::abs(edgeCoordinates[1] - edgeCoordinates[0]);
        }

        if (edgeCoordinates[2] >= 0 && edgeCoordinates[3] >= 0) {
            horizontalDistance = std::abs(edgeCoordinates[3] - edgeCoordinates[2]);
        }

        // 16. 计算中心位置
        if (horizontalDistance > 0 && verticalDistance > 0) {
            bestLoc.x = edgeCoordinates[2] + (horizontalDistance / 2.0f);
            bestLoc.y = edgeCoordinates[0] + (verticalDistance / 2.0f);
            return true;
        }
        return false;
    }
    catch (const std::exception& e) {
        return false;
    }
}

