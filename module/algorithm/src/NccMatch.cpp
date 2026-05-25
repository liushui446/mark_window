#include "NccMatch.h"
#include <limits>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // 通常和图像处理函数一起包含
#include "StegerSubpixel.h"
#include "icp.h"
#include "core/core.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

#define DRAW_MATCH_EDGE 1
#define CV_PI_RAD (CV_PI / 180.0f)
NccMatch::NccMatch(int angleStep, int minAngle, int maxAngle,
    double cannyThresh1, double cannyThresh2,
    double contourAreaThresh)
    : angleStep_(angleStep), minAngle_(minAngle), maxAngle_(maxAngle),
    cannyThresh1_(cannyThresh1), cannyThresh2_(cannyThresh2),
    cannyApertureSize_(3), cannyL2gradient_(true),
    contourAreaThresh_(contourAreaThresh) {}



bool NccMatch::extractEdgePoints(const cv::Mat& grayImage, std::vector<cv::Point>& edgePoints) {
    // 边缘检测
    cv::Mat edges, binary;

    //对于暗的图像提取效果不好
    //cv::Mat blurred;
    //cv::GaussianBlur(grayImage, blurred, cv::Size(3, 3), 1.0);
    //cv::Mat dummy;
    //double otsu = cv::threshold(blurred, dummy, 0, 255,
    //    cv::THRESH_BINARY | cv::THRESH_OTSU);
    //double highThresh = otsu;
    //double lowThresh = 0.4 * otsu;

    //cv::Canny(blurred, edges, lowThresh, highThresh, 3, true);

    cv::Scalar mean, stddev;
    cv::meanStdDev(grayImage, mean, stddev);
    const double imgStd = stddev[0];
    const bool lowContrast = (imgStd < 15.0);

    // ---- 保边降噪（bilateralFilter 保留真实边缘，对模板质量至关重要）----
    const double sigmaColor = std::max(15.0, 3.0 * imgStd);
    cv::Mat denoised;
    cv::GaussianBlur(grayImage, denoised, cv::Size(3, 3), 1.0);

    // ---- 自适应 Canny 阈值 ----
    double lowT, highT;
    if (lowContrast) {
        lowT = 20.0; highT = 50.0;
    }
    else {
        cv::Mat dummy;
        const double otsu = cv::threshold(denoised, dummy, 0, 255,
            cv::THRESH_BINARY | cv::THRESH_OTSU);
        highT = otsu;
        lowT = 0.4 * otsu;
    }
    cv::Canny(denoised, edges, lowT, highT, 3, /*L2gradient=*/true);

    lastEdgeImage_ = edges;  // 不需要 clone，下面就不再写 edges

    // ---- 连通域过滤 + 边缘点提取（一次扫描搞定）----
    const int minArea = lowContrast ? 30 : 20;
    extractEdgePointsWithNoiseFilter(edges, edgePoints, minArea);

    // ---- 兜底：完全没提到点就放宽阈值再试一次 ----
    if (edgePoints.empty()) {
        cv::Canny(denoised, edges, 10, 30, 3, true);
        //cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, k3);
        lastEdgeImage_ = edges;
        extractEdgePointsWithNoiseFilter(edges, edgePoints, 15);
    }

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
bool NccMatch::buildAngleTemplates(const std::vector<cv::Point2f>& edgePoints,
    const std::vector<cv::Point2f>& subedgePoints) {
    // 清空状态
    spatialTemplates_.clear();
    template_pyrdown_.clear();
    rotatedSubedgePoints_.clear();
    alignedTemplates_.clear();
    baseEdgePoints_.clear();
    baseSubedgePoints_.clear();
    fourierTemplates_.clear();
    lastRefSize_ = cv::Size();
    templatesBuilt_ = false;

    if (edgePoints.empty() || edgePoints.size() != subedgePoints.size()) return false;

    const int nAngles = (maxAngle_ - minAngle_) / angleStep_ + 1;

    // 基础包围框
    baseBbox_ = cv::boundingRect(edgePoints);
    baseW_ = baseBbox_.width;
    baseH_ = baseBbox_.height;
    if (baseW_ < 20 || baseH_ < 20) return false;

    // 小模板标记：宽或高 < 500 时粗匹配不做下采样
    smallTemplate_ = (baseW_ < 500 || baseH_ < 500);

    const float cx = baseW_ * 0.5f;
    const float cy = baseH_ * 0.5f;
    origTempCenter_ = { cx, cy };

    // 平移到基坐标系
    baseEdgePoints_.reserve(edgePoints.size());
    baseSubedgePoints_.reserve(subedgePoints.size());
    for (size_t i = 0; i < edgePoints.size(); ++i) {
        int px = cvRound(edgePoints[i].x - baseBbox_.x);
        int py = cvRound(edgePoints[i].y - baseBbox_.y);
        if ((unsigned)px <= (unsigned)baseW_ && (unsigned)py <= (unsigned)baseH_)
            baseEdgePoints_.emplace_back(px, py);

        float sx = subedgePoints[i].x - baseBbox_.x;
        float sy = subedgePoints[i].y - baseBbox_.y;
        baseSubedgePoints_.emplace_back(sx, sy);
    }

    // 0 层基底
    cv::Mat baseTemp = cv::Mat::zeros(baseH_, baseW_, CV_8UC1);
    for (const auto& p : baseEdgePoints_) baseTemp.ptr<uchar>(p.y)[p.x] = 255;

    // 同步到全局 core
    sm::Core* core = sm::Core::get_init();
    core->features.clear();
    core->features.reserve(baseSubedgePoints_.size());
    for (const auto& s : baseSubedgePoints_) {
        Point_f pt;
        pt.x = s.x - cx + 0.5f;
        pt.y = s.y - cy + 0.5f;
        core->features.push_back(pt);
    }

    // 2 层（4× 下采样）基底
    const cv::Point2f baseCenter(baseBbox_.x + baseW_ * 0.5f,
        baseBbox_.y + baseH_ * 0.5f);
    std::vector<cv::Point> scaled2EdgePoints;
    scaled2EdgePoints.reserve(edgePoints.size());
    for (const auto& p : edgePoints) {
        float dx = (p.x - baseCenter.x) * 0.25f;
        float dy = (p.y - baseCenter.y) * 0.25f;
        scaled2EdgePoints.emplace_back(cvRound(baseCenter.x + dx),
            cvRound(baseCenter.y + dy));
    }
    cv::Rect s2Bbox = cv::boundingRect(scaled2EdgePoints);
    const int s2W = s2Bbox.width, s2H = s2Bbox.height;
    if (s2W < 5 || s2H < 5) return false;

    cv::Mat scaled2BaseTemp = cv::Mat::zeros(s2H, s2W, CV_8UC1);
    for (const auto& p : scaled2EdgePoints)
        scaled2BaseTemp.ptr<uchar>(p.y - s2Bbox.y)[p.x - s2Bbox.x] = 255;
    const cv::Point2f s2Center(s2W * 0.5f, s2H * 0.5f);

    // 预算 2 层旋转矩阵
    std::vector<cv::Mat>  s2RotMats(nAngles);
    std::vector<cv::Size> s2RotSizes(nAngles);
    for (int i = 0; i < nAngles; ++i) {
        const int angle = minAngle_ + i * angleStep_;
        cv::RotatedRect rr(s2Center, cv::Size(s2W, s2H), angle);
        cv::Rect bbox = rr.boundingRect();
        cv::Mat M = cv::getRotationMatrix2D(s2Center, angle, 1.0);
        M.at<double>(0, 2) += bbox.width * 0.5 - s2Center.x;
        M.at<double>(1, 2) += bbox.height * 0.5 - s2Center.y;
        s2RotMats[i] = M;
        s2RotSizes[i] = bbox.size();
    }

    // 预分配输出，避免 push_back + critical
    template_pyrdown_.assign(nAngles, {});
    rotatedSubedgePoints_.assign(nAngles, {});
    std::vector<cv::Mat> tempSpatialTemps(nAngles);

    // ★ 真正并行：所有写入按索引，零 critical
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nAngles; ++i) {
        const int angle = minAngle_ + i * angleStep_;
        const float angleRad = float(angle) * float(CV_PI / 180.0);

        // 0 层旋转
        cv::RotatedRect rr(origTempCenter_, cv::Size(baseW_, baseH_), angle);
        cv::Rect bbox = rr.boundingRect();
        cv::Mat M = cv::getRotationMatrix2D(origTempCenter_, angle, 1.0);
        M.at<double>(0, 2) += bbox.width * 0.5 - origTempCenter_.x;
        M.at<double>(1, 2) += bbox.height * 0.5 - origTempCenter_.y;
        const double a00 = M.at<double>(0, 0), a01 = M.at<double>(0, 1), a02 = M.at<double>(0, 2);
        const double a10 = M.at<double>(1, 0), a11 = M.at<double>(1, 1), a12 = M.at<double>(1, 2);

        cv::Mat origRotated = (angle == 0)
            ? baseTemp                                       // 浅拷贝复用
            : cv::Mat::zeros(bbox.size(), CV_8UC1);

        if (angle != 0) {
            for (const auto& pt : baseEdgePoints_) {
                int x = int(a00 * pt.x + a01 * pt.y + a02 + 0.5);
                int y = int(a10 * pt.x + a11 * pt.y + a12 + 0.5);
                if ((unsigned)x < (unsigned)bbox.width &&
                    (unsigned)y < (unsigned)bbox.height)
                    origRotated.ptr<uchar>(y)[x] = 255;
            }
        }

        std::vector<cv::Point2f> rotatedSubPts;
        rotatedSubPts.reserve(baseSubedgePoints_.size());
        for (const auto& sp : baseSubedgePoints_) {
            float x = float(a00 * sp.x + a01 * sp.y + a02);
            float y = float(a10 * sp.x + a11 * sp.y + a12);
            if (x >= 0 && x < bbox.width && y >= 0 && y < bbox.height)
                rotatedSubPts.emplace_back(x, y);
        }

        // 2 层旋转
        cv::Mat scaled2Rotated;
        cv::warpAffine(scaled2BaseTemp, scaled2Rotated,
            s2RotMats[i], s2RotSizes[i],
            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

        template_pyrdown_[i] = { { origRotated, scaled2Rotated }, angleRad };
        rotatedSubedgePoints_[i] = { std::move(rotatedSubPts), angle };
        // 小模板用 0 层（原始尺寸）做粗匹配，大模板用 2 层（下采样）
        tempSpatialTemps[i] = smallTemplate_ ? origRotated : scaled2Rotated;
    }

    // 对齐到最大尺寸
    int maxR = 0, maxC = 0;
    for (const auto& t : tempSpatialTemps) {
        maxR = std::max(maxR, t.rows);
        maxC = std::max(maxC, t.cols);
    }
    maxTemplateSize_ = { maxC, maxR };
    if (maxTemplateSize_.area() == 0) return false;

    alignedTemplates_.assign(nAngles, cv::Mat());
    const int rowBase = maxR >> 1, colBase = maxC >> 1;
    for (int i = 0; i < nAngles; ++i) {
        const auto& t = tempSpatialTemps[i];
        alignedTemplates_[i] = cv::Mat::zeros(maxTemplateSize_, t.type());
        const int rOff = rowBase - (t.rows >> 1);
        const int cOff = colBase - (t.cols >> 1);
        t.copyTo(alignedTemplates_[i](cv::Rect(cOff, rOff, t.cols, t.rows)));
    }

    spatialTemplates_.reserve(nAngles);
    for (int i = 0; i < nAngles; ++i)
        spatialTemplates_.emplace_back(alignedTemplates_[i], template_pyrdown_[i].second);

    templatesBuilt_ = true;
    return true;
}

// ---- 只跟测试图尺寸有关；同尺寸图直接复用 ----
bool NccMatch::prepareForImage(const cv::Mat& refImage) {
    if (!templatesBuilt_ || alignedTemplates_.empty()) return false;
    if (refImage.size() == lastRefSize_ && !fourierTemplates_.empty())
        return true;   // 同尺寸图直接复用

    fourierTemplates_.clear();

    if (smallTemplate_) {
        // 小模板：不下采样，直接用原始分辨率
        downsampledRefImage_ = refImage;  // 保存引用，matchAngleTemplate 会复用
        corrSize_ = { refImage.cols - maxTemplateSize_.width + 1,
                      refImage.rows - maxTemplateSize_.height + 1 };

        fourierTemplates_.reserve(alignedTemplates_.size());
        for (auto& t : alignedTemplates_) {
            cv::Mat dftT;
            dftTemp(refImage, t, dftT, CV_32F);
            if (!dftT.empty()) fourierTemplates_.push_back(std::move(dftT));
        }
    }
    else {
        // 大模板：4 倍下采样
        cv::pyrDown(refImage, downsampledRefImage_,
            cv::Size(refImage.cols / 2, refImage.rows / 2));
        cv::pyrDown(downsampledRefImage_, downsampledRefImage_,
            cv::Size(downsampledRefImage_.cols / 2,
                downsampledRefImage_.rows / 2));

        corrSize_ = { downsampledRefImage_.cols - maxTemplateSize_.width + 1,
                      downsampledRefImage_.rows - maxTemplateSize_.height + 1 };

        fourierTemplates_.reserve(alignedTemplates_.size());
        for (auto& t : alignedTemplates_) {
            cv::Mat dftT;
            dftTemp(downsampledRefImage_, t, dftT, CV_32F);
            if (!dftT.empty()) fourierTemplates_.push_back(std::move(dftT));
        }
    }
    lastRefSize_ = refImage.size();
    return !fourierTemplates_.empty();
}

void NccMatch::robustCanny(const cv::Mat& gray, cv::Mat& edges) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    const double imgStd = stddev[0];
    const bool lowContrast = (imgStd < 15.0);

    const double sigmaColor = std::max(15.0, 3.0 * imgStd);
    cv::Mat denoised;
    cv::bilateralFilter(gray, denoised, 7, sigmaColor, 7);

    double lowT = 20.0, highT = 50.0;
    if (!lowContrast) {
        cv::Mat dummy;
        const double otsu = cv::threshold(denoised, dummy, 0, 255,
            cv::THRESH_BINARY | cv::THRESH_OTSU);
        highT = otsu;
        lowT = 0.4 * otsu;
    }
    cv::Canny(denoised, edges, lowT, highT, 3, true);

    // 兜底：完全没边再用增强路线
    if (cv::countNonZero(edges) == 0) {
        cv::Mat tmp1, tmp2;
        cv::medianBlur(gray, tmp1, 3);
        cv::equalizeHist(tmp1, tmp1);
        cv::bilateralFilter(tmp1, tmp2, 9, 75, 75);
        cv::Canny(tmp2, edges, 170, 230, 3, true);
    }
}
bool NccMatch::generateTemplateAndSaveEdgePoints(const cv::Mat& grayImage, const std::string& edgeXmlPath) {
    std::vector<cv::Point> edgePoints;
    if (!extractEdgePoints(grayImage, edgePoints)) {
        return false;
    }

    // 亚像素提取
    std::vector<cv::Point2f> subedgePoints;
    subedgePoints.reserve(edgePoints.size());
    SubPixelByZernike1(grayImage, edgePoints, subedgePoints);

    // 写入全局 core
    sm::Core* core = sm::Core::get_init();
    core->sub_features.clear();
    core->sub_features.reserve(subedgePoints.size());
    for (const auto& p : subedgePoints) {
        core->sub_features.emplace_back(Point_f{ p.x, p.y });
    }

    std::vector<cv::Point2f> edgePoints2f;
    edgePoints2f.reserve(edgePoints.size());
    for (const auto& p : edgePoints)
        edgePoints2f.emplace_back(float(p.x), float(p.y));

    return buildAngleTemplates(edgePoints2f, subedgePoints);
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
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    double bestVal = std::numeric_limits<double>::max();
    result = 0;
    for (unsigned int i = 0; i < image.size(); i++)
    {
        if (image[i].empty()) continue;
        cv::minMaxLoc(image[i], &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        // 距离变换匹配：模板边缘对齐图像边缘时互相关值最低，选 minVal 最小的角度
        if (minVal < bestVal) {
            bestVal = minVal;
            result = i;
        }
    }
    return true;
}
bool NccMatch::matchAngleTemplate(const cv::Mat& grayImage, std::vector<cv::Mat>& results) {
    if (template_pyrdown_.empty()) return false;

    cv::Mat match_edge;
    robustCanny(grayImage, match_edge);

    cv::Mat edge_inv;
    cv::bitwise_not(match_edge, edge_inv);
    cv::Mat dist = cv::Mat::zeros(grayImage.size(), CV_32FC1);
    cv::distanceTransform(edge_inv, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    if (smallTemplate_) {
        // 小模板：用 cv::matchTemplate + TM_CCOEFF_NORMED，归一化不受模板大小影响
        cv::exp(-0.2 * dist, dist);
        cv::Mat distScore = 1.0f - dist;

        results.resize(template_pyrdown_.size());
        for (size_t i = 0; i < template_pyrdown_.size(); ++i) {
            cv::Mat templFloat;
            template_pyrdown_[i].first[0].convertTo(templFloat, CV_32F, 1.0 / 255.0);
            cv::matchTemplate(distScore, templFloat, results[i], cv::TM_CCOEFF_NORMED);
        }
        return true;
    }

    // 大模板：原有 FFT 流程
    if (fourierTemplates_.empty()) return false;
    cv::Mat matchImage;
    if (!downsampledRefImage_.empty() &&
        downsampledRefImage_.cols == grayImage.cols / 4 &&
        downsampledRefImage_.rows == grayImage.rows / 4) {
        matchImage = downsampledRefImage_;
    }
    else {
        cv::pyrDown(grayImage, matchImage,
            cv::Size(grayImage.cols / 2, grayImage.rows / 2));
        cv::pyrDown(matchImage, matchImage,
            cv::Size(matchImage.cols / 2, matchImage.rows / 2));
    }

    cv::Mat big_edge;
    robustCanny(matchImage, big_edge);
    cv::Mat big_inv;
    cv::bitwise_not(big_edge, big_inv);
    cv::Mat bigDist = cv::Mat::zeros(matchImage.size(), CV_32FC1);
    cv::distanceTransform(big_inv, bigDist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    results.resize(fourierTemplates_.size());
    return matchFttTemplate(bigDist, fourierTemplates_, corrSize_, results, cv::TM_CCORR);
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
bool NccMatch::loadEdgePointsAndGenerateTemplates(const std::string& edgeXmlPath,
    const cv::Mat& Image) {
    (void)edgeXmlPath;

    // 模板已经在 NCC_CreateTemplate 里建好
    if (templatesBuilt_) return prepareForImage(Image);

    // 兜底：如果走异常路径直接进了匹配（同一进程内未建模板），
    // 从 core 恢复一次。这条路径在正常调用顺序下不会走到。
    sm::Core* core = sm::Core::get_init();
    if (core->temp_features.empty() || core->sub_features.empty()) {
        std::cerr << "templates not built and core has no features" << std::endl;
        return false;
    }
    std::vector<cv::Point2f> ep, sp;
    ep.reserve(core->temp_features.size());
    sp.reserve(core->sub_features.size());
    for (const auto& p : core->temp_features) ep.emplace_back(p.x, p.y);
    for (const auto& p : core->sub_features)  sp.emplace_back(p.x, p.y);
    if (!buildAngleTemplates(ep, sp)) return false;
    return prepareForImage(Image);
}
// 实现完整匹配流程函数（含边缘点加载）多角度
bool NccMatch::runFullMatchingFromPath(const cv::Mat& grayImage,
    const std::string& edgeXmlPath,
    cv::Point2f& bestLoc,
    float& bestAngle,
    double& bestScore)
{
    bestLoc = { -1, -1 };
    bestAngle = 0.0f;
    bestScore = 0.0;  // 注意：初始化为 0（无匹配），而非 1.0

    try {
        // 1) 确保灰度（避免无谓 clone）
        cv::Mat testGray;
        if (grayImage.channels() == 1)        testGray = grayImage;  // 不 clone
        else if (grayImage.channels() == 3)   cv::cvtColor(grayImage, testGray, cv::COLOR_BGR2GRAY);
        else                                  return false;

        // 2) 准备模板（首次完整建，之后零成本/小成本）
       if (!loadEdgePointsAndGenerateTemplates(edgeXmlPath, testGray)) return false;

        // 3) 粗匹配（小图角度搜索）
        std::vector<cv::Mat> coarseResults;
        if (!matchAngleTemplate(testGray, coarseResults)) return false;

        int coarseIdx = 0;
        bestTemplate(coarseResults, cv::TM_CCORR, coarseIdx);

        // 4) 全分辨率边缘 + 距离变换
        cv::Mat image_edge;
        robustCanny(testGray, image_edge);

        cv::Mat edge_inv;
        cv::bitwise_not(image_edge, edge_inv);

        cv::Mat dist;
        cv::distanceTransform(edge_inv, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        // 距离变换评分：1.0 - exp(-0.2 * dist)
        cv::exp(-0.2 * dist, dist);
        cv::Mat distScore = 1.0f - dist;

        // 5) 精匹配（单角度）
        cv::Mat& rotTempl = template_pyrdown_[coarseIdx].first[0];
        std::vector<cv::Mat> selected{ rotTempl };
        std::vector<cv::Mat> fineResults(1);
        matchNCCTemplate(distScore, selected, fineResults, cv::TM_CCORR_NORMED);

        // 6) 解析最佳位置
        std::vector<std::pair<double, cv::Point2f>> preciseLoc;
        double preciseVal = 0.0;
        matchLocation(fineResults[0], cv::TM_CCORR_NORMED, preciseLoc,
            rotTempl.cols, rotTempl.rows, preciseVal);
        if (preciseLoc.empty()) return false;
        bestLoc = preciseLoc[0].second;
        bestScore = preciseLoc[0].first;

        // 7) ICP 亚像素精细化（稀疏采样加速，精度损失极小）
        const auto& rotSubPts = rotatedSubedgePoints_[coarseIdx].first;
        sm::imgproc::Scene_edge scene;
        std::vector<sm::imgproc::Vec2f> pcdBuf, normalBuf;
        scene.init_Scene_edge(testGray, pcdBuf, normalBuf);

        // 均匀稀疏采样：限制最多 800 点，减少 ICP 迭代开销
        const size_t totalPts = rotSubPts.size();
        const size_t maxPts = 800;
        const size_t step = (totalPts > maxPts) ? (totalPts + maxPts - 1) / maxPts : 1;

        std::vector<sm::imgproc::Vec2f> modelPcd;
        modelPcd.reserve((totalPts + step - 1) / step);
        const float dx = bestLoc.x - rotTempl.cols * 0.5f;
        const float dy = bestLoc.y - rotTempl.rows * 0.5f;
        for (size_t i = 0; i < totalPts; i += step) {
            const auto& p = rotSubPts[i];
            modelPcd.push_back({ p.x + dx, p.y + dy });
        }

        auto Re = sm::imgproc::icp::ICP2D_Point2Plane(modelPcd, scene);
        const float xRef = float(Re.transformation_[0][0]) * dx
            + float(Re.transformation_[0][1]) * dy
            + float(Re.transformation_[0][2]);
        const float yRef = float(Re.transformation_[1][0]) * dx
            + float(Re.transformation_[1][1]) * dy
            + float(Re.transformation_[1][2]);
        const float rDeg = std::atan2(float(Re.transformation_[1][0]),
            float(Re.transformation_[0][0])) * 180.0f / float(CV_PI);

        // 8) 最终位置/角度
        bestLoc.x = xRef + rotTempl.cols * 0.5f;
        bestLoc.y = yRef + rotTempl.rows * 0.5f;
        const int sumIndex = coarseIdx + 1;  // 已无 fine angle，单角度
        bestAngle = (sumIndex == 0) ? (-5.0f - rDeg) : float(sumIndex - 6) - rDeg;

#ifdef DRAW_MATCH_EDGE
        // 用旋转后的亚像素点直接画，O(N) 而非 O(W·H)
        const float ox = bestLoc.x - rotTempl.cols * 0.5f;
        const float oy = bestLoc.y - rotTempl.rows * 0.5f;
        for (const auto& p : rotSubPts) {
            cv::Point ip(cvRound(p.x + ox), cvRound(p.y + oy));
            if ((unsigned)ip.x < (unsigned)grayImage.cols &&
                (unsigned)ip.y < (unsigned)grayImage.rows) {
                // 这里用户自己的可视化 Mat，不展开
                cv::line(testGray,
                    cv::Point(ip.x, ip.y),
                    cv::Point(ip.x, ip.y),
                    cv::Scalar(255, 0, 0), 1);
            }
        }
#endif
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

bool NccMatch::runFullMatchingFromPath_zero(
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
        bestScore =1.0;
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
        std::vector<cv::Mat> selected_templ_dft;
        for (int i = 0; i < 3; i++)
        {
            int index1 = 4+i ;
            const auto& tempGroup = template_pyrdown_[index1];
            cv::Mat rotate_template_image = tempGroup.first[0];
            selected_templ_dft.push_back(rotate_template_image);
        }
        std::vector<cv::Mat> match_result(selected_templ_dft.size());
        matchNCCTemplate(image_edge_distance, selected_templ_dft, match_result, match_method);
        int best_template_index;
        bestTemplate(match_result, match_method, best_template_index);
        cv::Mat rotate_template_image = selected_templ_dft.at(best_template_index);
        
        std::vector<std::pair<double, cv::Point2f>> precise_location;
        double precise_val = 0.0;
        matchLocation(match_result.at(best_template_index), match_method, precise_location, rotate_template_image.cols, rotate_template_image.rows, precise_val);

        //位置计算
        bestLoc.x = precise_location[0].second.x;
        bestLoc.y = precise_location[0].second.y;
        //亚像素匹配
        vector<Point2f> rotated_points = rotatedSubedgePoints_[5].first;
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
            float dx = pt.x + (bestLoc.x - rotate_template_image.cols / 2);
            float dy = pt.y + (bestLoc.y - rotate_template_image.rows / 2);

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
        bestLoc.x = x_refine + rotate_template_image.cols / 2;
        bestLoc.y = y_refine + rotate_template_image.rows / 2;
        bestAngle = best_template_index - 1- r_deg;
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
void NccMatch::extractEdgePointsWithNoiseFilter(
    const cv::Mat& edgeImage,
    std::vector<cv::Point>& edgePoints,
    const int& minArea)
{
    edgePoints.clear();
    if (edgeImage.empty()) return;

    cv::Mat labels, stats, centroids;
    const int numLabels = cv::connectedComponentsWithStats(
        edgeImage, labels, stats, centroids, 8, CV_32S);
    if (numLabels <= 1) return;  // 只有背景

    // 第一步：找最大连通域面积
    int maxArea = 0;
    for (int i = 1; i < numLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > maxArea) maxArea = area;
    }
    if (maxArea < minArea) return;

    // 第二步：构造保留表
    // 策略：保留面积 >= minArea 且 >= 最大面积 5% 的连通域
    // 保留 5% 比例是为了保留目标的内外轮廓（如 box-in-box 的内框）
    // 而杂点通常只有最大面积的 0.1%~2%
    const int areaThreshold = std::max(minArea, maxArea / 20);
    std::vector<uchar> keep(numLabels, 0);
    int totalKeptPixels = 0;
    for (int i = 1; i < numLabels; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= areaThreshold) {
            keep[i] = 1;
            totalKeptPixels += area;
        }
    }
    if (totalKeptPixels == 0) return;

    // 预分配，避免反复扩容
    edgePoints.reserve(totalKeptPixels);
    sm::Core* core = sm::Core::get_init();
    core->temp_features.clear();
    core->temp_features.reserve(totalKeptPixels);

    // 缓存忽略区域指针，避免循环内反复访问
    const auto& ignoreRegions = core->ignore_regions;
    const size_t numIgnore = ignoreRegions.size();

    // 单遍扫描，行指针访问
    const int H = edgeImage.rows;
    const int W = edgeImage.cols;
    for (int y = 0; y < H; ++y) {
        const int* rowL = labels.ptr<int>(y);
        for (int x = 0; x < W; ++x) {
            const int l = rowL[x];
            if (l > 0 && keep[l]) {
                // 检查是否在忽略区域内
                bool ignored = false;
                for (size_t k = 0; k < numIgnore; ++k) {
                    const auto& ir = ignoreRegions[k];
                    if (x >= ir.x && x < ir.x + ir.width &&
                        y >= ir.y && y < ir.y + ir.height) {
                        ignored = true;
                        break;
                    }
                }
                if (ignored) continue;

                edgePoints.emplace_back(x, y);
                Point_f pt;
                pt.x = static_cast<float>(x);
                pt.y = static_cast<float>(y);
                core->temp_features.emplace_back(pt);
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
    double& bestScore,
    int markType
) {
        // 1. 初始化输出参数
        bestLoc = cv::Point2f(-1, -1);
        bestAngle = 0.0f;
        bestScore = 1.0;
        std::vector<cv::Rect> all_regions;
        double distanceX=0;
        double distanceY=0;
        bool success;
        cv::Mat testImg = grayImage.clone();
        std::vector<std::vector<double>> vecMagn;
        switch(markType) {
        case 0:  // Cross Mark
            // 针对十字标记的处理
            // ...
            break;
        case 1:  // Bar-in-Bar
            // 针对条形标记的处理
            //增加图像转正的方法

            success = Roi_extract(testImg,all_regions);
            if (!success) {
                return false;
            }
            distanceX = ExOntoTest(testImg, vecMagn, all_regions);
            cv::rotate(testImg, testImg, cv::ROTATE_90_CLOCKWISE);
            success = Roi_extract(testImg, all_regions);
            distanceY = ExOntoTest(testImg, vecMagn, all_regions);
            break;
        case 2:  // Box-in-Box
            // 针对盒中盒标记的处理
            // ...
            break;
        case 3:  // Frame-in-Frame
            // 针对帧中帧标记的处理
            // ...
            break;
        case 4:  // AIM (Advanced Imaging Mark)
            // 针对AIM标记的处理
            // ...
            success = AIM_Roi_extract(testImg, all_regions);
            if (!success) {
                return false;
            }
            distanceX = ExOntoTest1(testImg, vecMagn, all_regions);
            cv::rotate(testImg, testImg, cv::ROTATE_90_CLOCKWISE);
            success = AIM_Roi_extract(testImg, all_regions);
            distanceY = ExOntoTest1(testImg, vecMagn, all_regions);
            break;
        default:
            // 默认处理
            // ...
            break;
        }
       
        
        // 3. 读取待匹配图像并转换为灰度图
       // cv::Mat testImg = cv::imread(testImagePath);
        //cv::Mat testImg = grayImage.clone();
        //std::vector<std::vector<double>> vecMagn;
        //double distanceX = NccMatch::ExOntoTest(testImg, vecMagn, all_regions);
        //cv::rotate(testImg, testImg, cv::ROTATE_90_CLOCKWISE);
        //double distance_y = NccMatch::ExOntoTest1(testImg, vecMagn, all_regions);
        bestLoc.x = distanceX;
        bestLoc.y = distanceY;
        return true;
}
double NccMatch::CaluateLine2Line(vector<Point2f> line1, vector<Point2f> line2)
{
    Vec4f line_para;
    fitLine(line2, line_para, DIST_L2, 0, 0.01, 0.01);

    double A = line_para[1] / line_para[0];
    double B = -1;
    double C = -B * line_para[3] - A * line_para[2];

    double sum = 0, sumd = 0;
    for (size_t i = 0; i < line1.size(); i++)
    {
        Point p = line1[i];

        double d = abs(A * p.x + B * p.y + C) / sqrt(A * A + B * B);
        sumd += d;
    }


    Vec4f line_para1;
    fitLine(line1, line_para1, DIST_L2, 0, 0.01, 0.01);

    double A1 = line_para1[1] / line_para1[0];
    double B1 = -1;
    double C1 = -B1 * line_para1[3] - A1 * line_para1[2];

    double  sumd1 = 0;
    for (size_t i = 0; i < line2.size(); i++)
    {
        Point p = line2[i];

        double d = abs(A1 * p.x + B1 * p.y + C1) / sqrt(A1 * A1 + B1 * B1);
        sumd1 += d;
    }



    return (sumd / line1.size() + sumd1 / line2.size()) * 0.5;
}

double  NccMatch::CaluateLine2LinePCA(const std::vector<cv::Point2f>& line1,
    const std::vector<cv::Point2f>& line2)
{
    if (line1.size() < 2 || line2.size() < 2) {
        return -1.0;
    }

    // 1. 使用PCA拟合直线
    auto fitLinePCA = [](const std::vector<cv::Point2f>& points,
        cv::Point2f& center, cv::Point2f& direction) {
            // 计算中心点
            center = cv::Point2f(0, 0);
            for (const auto& pt : points) {
                center.x += pt.x;
                center.y += pt.y;
            }
            center.x /= points.size();
            center.y /= points.size();

            // 构建协方差矩阵
            double xx = 0, xy = 0, yy = 0;
            for (const auto& pt : points) {
                double dx = pt.x - center.x;
                double dy = pt.y - center.y;
                xx += dx * dx;
                xy += dx * dy;
                yy += dy * dy;
            }

            xx /= points.size();
            xy /= points.size();
            yy /= points.size();

            // 计算特征向量（主方向）
            double d = sqrt((xx - yy) * (xx - yy) + 4 * xy * xy);
            double lambda1 = (xx + yy + d) / 2;
            double lambda2 = (xx + yy - d) / 2;

            // 最大特征值对应的特征向量
            if (fabs(xy) > 1e-10) {
                direction.x = lambda1 - yy;
                direction.y = xy;
            }
            else {
                if (xx > yy) {
                    direction.x = 1;
                    direction.y = 0;
                }
                else {
                    direction.x = 0;
                    direction.y = 1;
                }
            }

            // 归一化
            double norm = sqrt(direction.x * direction.x + direction.y * direction.y);
            if (norm > 1e-10) {
                direction.x /= norm;
                direction.y /= norm;
            }
    };

    // 2. 拟合两条直线
    cv::Point2f center1, dir1, center2, dir2;
    fitLinePCA(line1, center1, dir1);
    fitLinePCA(line2, center2, dir2);

    // 3. 计算两条直线间的距离
    // 方法1: 使用向量投影法（假设两条线平行）
    cv::Point2f vecBetweenCenters = center2 - center1;

    // 计算垂直于直线的单位法向量
    cv::Point2f normal(-dir1.y, dir1.x);
    double normNormal = sqrt(normal.x * normal.x + normal.y * normal.y);
    if (normNormal > 1e-10) {
        normal.x /= normNormal;
        normal.y /= normNormal;
    }

    // 两个中心点在法向量方向上的投影距离
    double distance = fabs(vecBetweenCenters.x * normal.x + vecBetweenCenters.y * normal.y);

    // 4. 验证直线是否大致平行
    double dotProduct = fabs(dir1.x * dir2.x + dir1.y * dir2.y);
    if (dotProduct < 0.9) {  // 如果夹角大于约25度
        std::cerr << "Warning: Lines are not parallel (dot product = " << dotProduct << ")" << std::endl;
    }

    return distance;
}

bool NccMatch::Roi_extract(
    const cv::Mat& grayImage,
    std::vector<cv::Rect>& regions
) {
    try {
        regions.clear();
        regions.resize(4);  // 内层左、内层右、外层左、外层右

        cv::Mat smoothedImage;
        cv::GaussianBlur(grayImage, smoothedImage, cv::Size(5, 5), 1.5, 1.5);

        cv::Mat image_edge;
        cv::Canny(smoothedImage, image_edge, 140, 200, 3, true);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(image_edge.clone(), contours, hierarchy,
            cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            return false;
        }

        std::vector<std::vector<cv::Point>> external_contours;
        std::vector<std::vector<cv::Point>> innermost_contours;

        for (int i = 0; i < contours.size(); i++) {
            if (hierarchy[i][3] == -1) {
                external_contours.push_back(contours[i]);
            }
            if (hierarchy[i][2] == -1) {
                innermost_contours.push_back(contours[i]);
            }
        }

        if (external_contours.empty()) {
            return false;
        }

        // 处理外部轮廓
        cv::Rect outer_left_rect, outer_right_rect;

        if (!external_contours.empty()) {
            std::vector<cv::Point> allExternalPoints;
            for (const auto& contour : external_contours) {
                allExternalPoints.insert(allExternalPoints.end(),
                    contour.begin(), contour.end());
            }

            cv::RotatedRect minRect = cv::minAreaRect(allExternalPoints);
            cv::Rect boundingRect = minRect.boundingRect();

            boundingRect &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);

            if (boundingRect.width > 0 && boundingRect.height > 0) {
                int rectX = boundingRect.x;
                int rectY = boundingRect.y;
                int rectWidth = boundingRect.width;
                int rectHeight = boundingRect.height;

                float edgeRatio = 0.4f;

                // 计算外部左右两个区域
                int leftRegionHeight = static_cast<int>(rectHeight * (1.0f - edgeRatio));
                int leftStartY = rectY + (rectHeight - leftRegionHeight) / 2;
                int leftWidth = 30;
                int leftX = rectX - 15;
                cv::Rect leftRect(leftX, leftStartY, leftWidth, leftRegionHeight);
                leftRect &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);

                int rightStartY = leftStartY;
                int rightWidth = 30;
                int rightX = (rectX + rectWidth) - 15;
                cv::Rect rightRect(rightX, rightStartY, rightWidth, leftRegionHeight);
                rightRect &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);

                // 检查提取的矩形是否有效
                if (leftRect.width > 0 && leftRect.height > 0 &&
                    rightRect.width > 0 && rightRect.height > 0) {
                    outer_left_rect = leftRect;
                    outer_right_rect = rightRect;
                }
                else {
                    return false;  // 外部ROI无效，直接返回false
                }
            }
            else {
                return false;  // 外接矩形无效，直接返回false
            }
        }
        else {
            return false;  // 没有外部轮廓，直接返回false
        }

        // 处理内部轮廓
        cv::Rect inner_left_rect, inner_right_rect;

        if (!innermost_contours.empty()) {
            int maxAreaIdx = 0;
            double maxArea = 0;
            for (int i = 0; i < innermost_contours.size(); i++) {
                double area = cv::contourArea(innermost_contours[i]);
                if (area > maxArea) {
                    maxArea = area;
                    maxAreaIdx = i;
                }
            }

            std::vector<cv::Point> innerContour = innermost_contours[maxAreaIdx];
            cv::RotatedRect innerMinRect = cv::minAreaRect(innerContour);
            cv::Rect innerBoundingRect = innerMinRect.boundingRect();

            innerBoundingRect &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);

            if (innerBoundingRect.width > 0 && innerBoundingRect.height > 0) {
                int innerRectX = innerBoundingRect.x;
                int innerRectY = innerBoundingRect.y;
                int innerRectWidth = innerBoundingRect.width;
                int innerRectHeight = innerBoundingRect.height;

                float edgeRatio = 0.4f;

                // 计算内部左右两个区域
                int innerLeftRegionHeight = static_cast<int>(innerRectHeight * (1.0f - edgeRatio));
                int innerLeftStartY = innerRectY + (innerRectHeight - innerLeftRegionHeight) / 2;
                int innerLeftWidth = 30;
                int innerLeftX = innerRectX - 15;
                cv::Rect innerLeftRect(innerLeftX, innerLeftStartY,
                    innerLeftWidth, innerLeftRegionHeight);
                innerLeftRect &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);

                int innerRightStartY = innerLeftStartY;
                int innerRightWidth = 30;
                int innerRightX = (innerRectX + innerRectWidth) - 15;
                cv::Rect innerRightRect(innerRightX, innerRightStartY,
                    innerRightWidth, innerLeftRegionHeight);
                innerRightRect &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);

                // 检查提取的矩形是否有效
                if (innerLeftRect.width > 0 && innerLeftRect.height > 0 &&
                    innerRightRect.width > 0 && innerRightRect.height > 0) {
                    inner_left_rect = innerLeftRect;
                    inner_right_rect = innerRightRect;
                }
                else {
                    return false;  // 内部ROI无效，直接返回false
                }
            }
            else {
                return false;  // 内接矩形无效，直接返回false
            }
        }
        else {
            return false;  // 没有内部轮廓，直接返回false
        }

        // 将ROI区域按顺序存储到regions向量中
        // 顺序：内层左、内层右、外层左、外层右
        regions[0] = inner_left_rect;
        regions[1] = inner_right_rect;
        regions[2] = outer_left_rect;
        regions[3] = outer_right_rect;

        // 检查是否成功提取了所有ROI区域
        for (const auto& rect : regions) {
            if (rect.width <= 0 || rect.height <= 0) {
                return false;  // 修改为直接返回false
            }
        }

        return true;  // 所有ROI都有效
    }
    catch (const std::exception& e) {
        return false;
    }
}

bool NccMatch::AIM_Roi_extract(
    const cv::Mat& grayImage,
    std::vector<cv::Rect>& regions
) {
    try {
        regions.clear();
        regions.resize(2);  // AIM标记：左上角矩形左边和右下角矩形右边，共2个ROI

        cv::Mat image_edge;
        
        // 图像预处理增强
            // 预处理：中值滤波去除椒盐噪声
        cv::Mat medianFilteredImage;
        cv::medianBlur(grayImage, medianFilteredImage, 3);
        cv::Mat preprocessedImage;
        cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

        // 使用双边滤波减少噪声
        cv::Mat smoothedImage, edges2;
        cv::bilateralFilter(preprocessedImage, smoothedImage, 9, 75, 75);
        //cv::Canny(smoothedImage2, edges2, 200, 230, 3, true);
        cv::Canny(smoothedImage, image_edge, 200, 200, 3, true);

        // 可选：形态学操作连接断裂边缘
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(image_edge, image_edge, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(image_edge.clone(), contours, hierarchy,
            cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  // 只提取外部轮廓

        if (contours.empty()) {
            return false;
        }

        // 寻找所有矩形轮廓
        std::vector<cv::Rect> candidate_rects;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);

            // 过滤面积太小的轮廓
            if (area < 20 ) {
                continue;
            }

            // 计算轮廓的边界矩形
            cv::Rect rect = cv::boundingRect(contours[i]);

            // 计算矩形宽高比，过滤过于细长的轮廓（AIM标记通常是接近正方形的矩形）
            double aspect_ratio = static_cast<double>(rect.width) / rect.height;
            //if (aspect_ratio < 0.7 || aspect_ratio > 1.3) {
            //    continue;
            //}

            // 计算轮廓的周长
            double perimeter = cv::arcLength(contours[i], true);
            double circularity = 4 * CV_PI * area / (perimeter * perimeter);

            // 筛选接近矩形的轮廓（圆形度在0.7-0.9之间）
            //if (circularity < 0.7 || circularity > 0.9) {
            //    continue;
            //}

            candidate_rects.push_back(rect);
        }

        if (candidate_rects.size() < 2) {
            return false;
        }

        // 按位置排序：先按y坐标（从上到下），再按x坐标（从左到右）
        std::sort(candidate_rects.begin(), candidate_rects.end(),
            [](const cv::Rect& a, const cv::Rect& b) {
                // 如果y坐标相差不大（在同一行），按x坐标排序
                if (abs(a.y - b.y) < 20) {
                    return a.x < b.x;
                }
                return a.y < b.y;
            });

        // 选择左上角和右下角的矩形
        // 左上角：排序后的第一个矩形
        cv::Rect top_left_rect = candidate_rects[0];

        // 右下角：需要找一个在右下角位置的矩形
        // 我们可以选择x+y最大的矩形
        cv::Rect bottom_right_rect = candidate_rects[0];
        int max_sum = 0;

        for (const auto& rect : candidate_rects) {
            int sum = rect.x + rect.y + rect.width + rect.height;
            if (sum > max_sum) {
                max_sum = sum;
                bottom_right_rect = rect;
            }
        }

        // 或者直接选择最后一个（如果排序正确的话）
        // cv::Rect bottom_right_rect = candidate_rects.back();

        // 调整ROI大小，确保ROI区域不会太小
        int min_roi_size = 15;  // 最小ROI尺寸

        // 计算左上角矩形左边的ROI
        int left_roi_width = std::max(30, min_roi_size);
        int left_roi_height = std::max(top_left_rect.height / 2, min_roi_size);
        int left_roi_x = top_left_rect.x - 20;  // 在矩形左边20像素处
        int left_roi_y = top_left_rect.y + top_left_rect.height / 4;  // 垂直居中

        cv::Rect left_roi(left_roi_x, left_roi_y, left_roi_width, left_roi_height);
        left_roi &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);  // 确保在图像范围内

        // 计算右下角矩形右边的ROI
        int right_roi_width = std::max(30, min_roi_size);
        int right_roi_height = std::max(bottom_right_rect.height / 2, min_roi_size);
        int right_roi_x = bottom_right_rect.x + bottom_right_rect.width - 10;  // 在矩形右边10像素处
        int right_roi_y = bottom_right_rect.y + bottom_right_rect.height / 4;

        cv::Rect right_roi(right_roi_x, right_roi_y, right_roi_width, right_roi_height);
        right_roi &= cv::Rect(0, 0, grayImage.cols, grayImage.rows);  // 确保在图像范围内

        // 检查ROI是否有效
        if (left_roi.width <= 0 || left_roi.height <= 0 ||
            right_roi.width <= 0 || right_roi.height <= 0) {
            return false;
        }

        // 检查ROI区域是否太小
        if (left_roi.area() < min_roi_size * min_roi_size ||
            right_roi.area() < min_roi_size * min_roi_size) {
            return false;
        }

        // 存储ROI区域
        regions[0] = left_roi;   // 左上角矩形左边ROI
        regions[1] = right_roi;  // 右下角矩形右边ROI

        // 可选：可视化调试（需要时启用）
        /*
        cv::Mat debug_img;
        cv::cvtColor(grayImage, debug_img, cv::COLOR_GRAY2BGR);

        // 绘制所有候选矩形
        for (const auto& rect : candidate_rects) {
            cv::rectangle(debug_img, rect, cv::Scalar(0, 255, 0), 1);
        }

        // 绘制选中的矩形
        cv::rectangle(debug_img, top_left_rect, cv::Scalar(255, 0, 0), 2);
        cv::rectangle(debug_img, bottom_right_rect, cv::Scalar(0, 0, 255), 2);

        // 绘制ROI区域
        cv::rectangle(debug_img, left_roi, cv::Scalar(0, 255, 255), 2);
        cv::rectangle(debug_img, right_roi, cv::Scalar(255, 255, 0), 2);

        cv::imshow("AIM ROI Extraction", debug_img);
        cv::waitKey(1);
        */

        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}
// 仿函数结构体：用于 Eigen 的非线性优化
struct AsymGaussFitting {
    const std::vector<double>& x_v;
    const std::vector<double>& y_v;
    const std::vector<double>& w_v;

    AsymGaussFitting(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& w)
        : x_v(x), y_v(y), w_v(w) {}

    // 计算残差 fvec
    int operator()(const Eigen::VectorXd& p, Eigen::VectorXd& fvec) const {
        double A = p(0);
        double mu = p(1);
        double sigL = p(2);
        double sigR = p(3);

        for (int i = 0; i < x_v.size(); ++i) {
            double diff = x_v[i] - mu;
            double sigma = (diff <= 0) ? sigL : sigR;
            double y_fit = A * std::exp(-(diff * diff) / (2.0 * sigma * sigma));
            // 强度加权残差
            fvec(i) = w_v[i] * (y_fit - y_v[i]);
        }
        return 0;
    }

    // 计算雅可比矩阵（导数），用于加速收敛
    int df(const Eigen::VectorXd& p, Eigen::MatrixXd& fjac) const {
        double eps = 1e-8;
        for (int j = 0; j < p.size(); j++) {
            Eigen::VectorXd p_plus = p; p_plus(j) += eps;
            Eigen::VectorXd p_minus = p; p_minus(j) -= eps;
            Eigen::VectorXd fvec_plus(values()), fvec_minus(values());
            (*this)(p_plus, fvec_plus);
            (*this)(p_minus, fvec_minus);
            fjac.col(j) = (fvec_plus - fvec_minus) / (2.0 * eps);
        }
        return 0;
    }

    int values() const { return static_cast<int>(x_v.size()); }
    int inputs() const { return 4; } // [A, mu, sigmaL, sigmaR]
};

double fitSingleLine(const std::vector<double>& raw_data) {
    // 1. 局部背景抑制
    double bg_level = *std::min_element(raw_data.begin(), raw_data.end());
    std::vector<double> y_proc(raw_data.size());
    double max_val = 0;
    int max_idx = 0;

    for (int i = 0; i < raw_data.size(); ++i) {
        y_proc[i] = std::max(0.0, raw_data[i] - bg_level);
        if (y_proc[i] > max_val) {
            max_val = y_proc[i];
            max_idx = i;
        }
    }

    // 2. 窄窗口策略 (±4个像素)
    int win_r = 4;
    std::vector<double> x_fit, y_fit, weights;
    for (int i = max_idx - win_r; i <= max_idx + win_r; ++i) {
        if (i >= 0 && i < y_proc.size()) {
            x_fit.push_back(static_cast<double>(i));
            y_fit.push_back(y_proc[i]);
            // 关键改进 C: 强度平方加权 (Intensity Weighting)

            double dist = std::abs(i - max_idx);
            double w_ = std::exp(-(dist * dist) / (2 * 1.0 * 1.0));
            //weights.push_back(w_);
            weights.push_back(std::pow(y_proc[i] / max_val, 2.0));
        }
    }

    // 3. 初始参数设置
    Eigen::VectorXd p(4);
    p << max_val, static_cast<double>(max_idx), 1.2, 1.2;

    // 4. 配置优化器 (对应 Matlab 1e-14 精度限制)
    AsymGaussFitting functor(x_fit, y_fit, weights);
    Eigen::LevenbergMarquardt<AsymGaussFitting> lm(functor);
    lm.parameters.ftol = 1e-14;
    lm.parameters.xtol = 1e-14;
    lm.parameters.maxfev = 2000;

    // 5. 执行拟合
    lm.minimize(p);

    return p(1); // 返回 mu (中心位置)
}


std::vector<double> computeDericheGradient(const std::vector<double>& input,
    double alpha,
    std::vector<double>& abs_gradients) {
    size_t L = input.size();
    if (L < 3) return std::vector<double>(L, 0.0);

    std::vector<double> y_plus(L, 0.0);
    std::vector<double> y_minus(L, 0.0);
    std::vector<double> gradients(L, 0.0);
    abs_gradients.assign(L, 0.0);

    // 1. 系数计算
    double exp_a = std::exp(-alpha);
    double b1 = -2.0 * exp_a;
    double b2 = std::exp(-2.0 * alpha);
    double k = std::pow(1.0 - exp_a, 2.0) / exp_a;
    double a = -k * exp_a;

    // 2. 因果滤波 (从左往右)
    // 核心改进：根据递归方程推导稳态初始值
    // 假设 m < 0 时，input[m] = input[0]
    // 此时 y_plus 的稳态平衡值应满足: y = input[0] - b1*y - b2*y
    double initial_y_plus = input[0] / (1.0 + b1 + b2);

    y_plus[0] = initial_y_plus;
    // y_plus[1] 依赖 y_plus[0] 和假设的 y_plus[-1]
    y_plus[1] = input[0] - b1 * y_plus[0] - b2 * initial_y_plus;

    for (size_t m = 2; m < L; ++m) {
        y_plus[m] = input[m - 1] - b1 * y_plus[m - 1] - b2 * y_plus[m - 2];
    }

    // 3. 非因果滤波 (从右往左)
    // 假设 m >= L 时，input[m] = input[L-1]
    double initial_y_minus = input[L - 1] / (1.0 + b1 + b2);

    y_minus[L - 1] = initial_y_minus;
    y_minus[L - 2] = input[L - 1] - b1 * y_minus[L - 1] - b2 * initial_y_minus;

    for (int m = static_cast<int>(L) - 3; m >= 0; --m) {
        y_minus[m] = input[m + 1] - b1 * y_minus[m + 1] - b2 * y_minus[m + 2];
    }

    // 4. 组合结果
    for (size_t m = 0; m < L; ++m) {
        // 这里的 a 已经包含了归一化因子
        double val = a * (y_plus[m] - y_minus[m]);
        gradients[m] = val;
        abs_gradients[m] = std::abs(val);
    }

    // 5. 强制首尾平滑化（可选）
    // 因为导数定义在中心，首尾像素缺乏完整邻域信息，通常设为0或邻近值
    gradients[0] = gradients[1] - 1e-6;
    gradients[L - 1] = gradients[L - 2] - 1e-6;
    abs_gradients[0] = abs_gradients[1] - 1e-6;
    abs_gradients[L - 1] = abs_gradients[L - 2] - 1e-6;

    return gradients;
}

struct GaussianParams {
    double A = 0.0;     // 峰值幅度
    double mu = 0.0;    // 亚像素位置（中心偏移）
    double sigma = 0.0; // 钟形曲线宽度
};

// 基础高斯函数：g(x) = A * exp(-(x-mu)^2 / (2*sigma^2))
inline double gaussian(double x, const GaussianParams& p) {
    return p.A * std::exp(-(std::pow(x - p.mu, 2)) / (2 * std::pow(p.sigma, 2)));
}

int ExtractEdge(cv::Mat src, cv::Rect roi, int direct, std::vector<cv::Point2f>& vec_edgePs)
{
    //沿着方向计算像素，并且计算梯度，寻找梯度最大值

    double alpha = 1.8;

    cv::Mat preImg;
    if (direct == 0)
    {
        preImg = src(roi);
    }
    else
    {
        preImg = src(roi);
    }

    cv::Mat mafnitudeImg(preImg.size(), CV_64F);
    cv::Mat idxImg(preImg.rows, 1, CV_64F);
    for (size_t i = 0; i < preImg.rows; i++)
    {
        std::vector<double> signal;
        for (size_t j = 0; j < preImg.cols; j++)
        {
            signal.push_back(preImg.at<uchar>(i, j));
        }
        std::vector<double> gradient, magnitude, stats;
        gradient = computeDericheGradient(signal, alpha, magnitude);

        double* row_ptr = mafnitudeImg.ptr<double>(i);
        for (int j = 0; j < mafnitudeImg.cols; ++j) {
            row_ptr[j] = magnitude[j];
        }



        std::vector<double>::iterator biggest = std::max_element(std::begin(magnitude), std::end(magnitude));
        //or std::vector<double>::iterator biggest = std::max_element(v.begin(), v.end);

        int idx = std::distance(std::begin(magnitude), biggest);

        std::vector<double> newMagn;
        for (int ll = -2; ll < 3; ll++)
        {
            int nIdx = idx + ll;
            if (nIdx > magnitude.size() - 1)
            {
                nIdx = magnitude.size() - 1;
            }
            if (nIdx < 0)
            {
                nIdx = 0;
            }

            newMagn.push_back(magnitude[nIdx]);
        }
        double dIdx = fitSingleLine(newMagn);
        idxImg.at<double>(i, 0) = dIdx;
        cv::Point2f center;
        center.x = dIdx + roi.x;
        center.y = i + roi.y;
        vec_edgePs.push_back(center);
    }
    return 0;
}


int ExtractEdge2(cv::Mat src, cv::Rect roi, int direct, std::vector<cv::Point2f>& vec_edgePs, std::vector<double>& magnitude)
{
    //沿着方向计算像素，并且计算梯度，寻找梯度最大值
    //列平均法

    double alpha = 1.8;

    cv::Mat preImg;
    if (direct == 0)
    {
        preImg = src(roi);
    }
    else
    {
        preImg = src(roi);
    }
    std::vector<double> signal;

    for (size_t i = 0; i < preImg.cols; i++)
    {
        double columnSum = 0.0;
        for (size_t j = 0; j < preImg.rows; j++)
        {
            columnSum = columnSum + preImg.at<uchar>(j, i);
        }
        signal.push_back(columnSum / (double)preImg.rows);

    }
    std::vector<double> gradient, stats;
    gradient = computeDericheGradient(signal, alpha, magnitude);

    cv::Mat magnitudeImg(1, preImg.cols, CV_64F);
    for (int j = 0; j < magnitudeImg.cols; ++j) {
        magnitudeImg.at<double>(0, j) = magnitude[j];
    }

    std::vector<double>::iterator biggest = std::max_element(std::begin(magnitude), std::end(magnitude));
    //or std::vector<double>::iterator biggest = std::max_element(v.begin(), v.end);

    int idx = std::distance(std::begin(magnitude), biggest);
    std::vector<double> newMagn;
    for (int ll = -4; ll < 5; ll++)
    {
        int nIdx = idx + ll;
        if (nIdx > magnitude.size() - 1)
        {
            nIdx = magnitude.size() - 1;
        }
        if (nIdx < 0)
        {
            nIdx = 0;
        }

        newMagn.push_back(magnitude[nIdx]);
    }

    double dIdx = fitSingleLine(newMagn) - 4 + idx;//idx; //computeSubpixelIndustrial(newMagn,2) + idx;

    for (size_t i = 0; i < preImg.rows; i++)
    {
        cv::Point2f center;
        center.x = dIdx + roi.x;
        center.y = i + roi.y;
        vec_edgePs.push_back(center);
    }


    return 0;
}

int CalculateCenter(std::vector<cv::Point2f> vec_edgePs1, std::vector<cv::Point2f> vec_edgePs2, int direct, cv::Point2f& center)
{
    cv::Point2f sumCenter(0, 0);
    for (size_t i = 0; i < vec_edgePs1.size(); i++)
    {
        sumCenter.x += vec_edgePs1[i].x;
        sumCenter.y += vec_edgePs1[i].y;
    }
    for (size_t i = 0; i < vec_edgePs2.size(); i++)
    {
        sumCenter.x += vec_edgePs2[i].x;
        sumCenter.y += vec_edgePs2[i].y;
    }
    center.x = sumCenter.x / (double)(vec_edgePs1.size() + vec_edgePs2.size());
    center.y = sumCenter.y / (double)(vec_edgePs1.size() + vec_edgePs2.size());

    return 0;
}
cv::Point2f findMinRectTopLeft(const cv::Mat& img) {
    // 在函数内部设置ROI
    cv::Rect roi(390, 584, 484, 484);
    cv::Mat img_roi = img(roi).clone();

    // Canny边缘检测
    cv::Mat image_edge;
    cv::Canny(img_roi, image_edge, 140, 200, 3, true);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image_edge.clone(), contours, hierarchy,
        cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 分离外部轮廓
    std::vector<cv::Point> allExternalPoints;
    for (int i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] == -1) {  // 外部轮廓
            allExternalPoints.insert(allExternalPoints.end(),
                contours[i].begin(), contours[i].end());
        }
    }

    // 计算最小外接矩形
    cv::Point2f topLeft(-1, -1);  // 默认无效坐标
    if (!allExternalPoints.empty()) {
        cv::RotatedRect minRect = cv::minAreaRect(allExternalPoints);

        // 获取最小外接矩形的边界矩形（非旋转）
        cv::Rect boundingRect = minRect.boundingRect();

        // 将ROI坐标转换回原图坐标
        topLeft = cv::Point2f(boundingRect.x + roi.x,
            boundingRect.y + roi.y);
    }

    return topLeft;
}
double NccMatch::ExOntoTest(cv::Mat img, std::vector<std::vector<double>>& vecMagn,std::vector<cv::Rect>& regions)
{
    std::vector<cv::Point2f> vec_inEdgePs1, vec_inEdgePs2, vec_outEdgePs1, vec_outEdgePs2;

    std::vector<double> magnitude1;
    cv::Rect r2(711, 550, 30, 80);
    ExtractEdge2(img, regions[0], 0, vec_inEdgePs1, magnitude1);
    cv::Rect r3(867, 550, 30, 80);
    std::vector<double> magnitude2;
    ExtractEdge2(img, regions[1], 0, vec_inEdgePs2, magnitude2);


    cv::Rect r1(628, 550, 30, 80);
    std::vector<double> magnitude3;
    ExtractEdge2(img, regions[2], 0, vec_outEdgePs1, magnitude3);
    cv::Rect r4(953, 550, 30, 80);
    std::vector<double> magnitude4;

    ExtractEdge2(img, regions[3], 0, vec_outEdgePs2, magnitude4);
    vecMagn.push_back(magnitude1);
    vecMagn.push_back(magnitude2);
    vecMagn.push_back(magnitude3);
    vecMagn.push_back(magnitude4);

    cv::Mat maImg(4, 30, CV_64F);
    double* row_ptr = maImg.ptr<double>(0);
    for (int j = 0; j < maImg.cols; ++j) {
        row_ptr[j] = magnitude1[j];
    }
    row_ptr = maImg.ptr<double>(1);
    for (int j = 0; j < maImg.cols; ++j) {
        row_ptr[j] = magnitude2[j];
    }
    row_ptr = maImg.ptr<double>(2);
    for (int j = 0; j < maImg.cols; ++j) {
        row_ptr[j] = magnitude3[j];
    }
    row_ptr = maImg.ptr<double>(3);
    for (int j = 0; j < maImg.cols; ++j) {
        row_ptr[j] = magnitude4[j];
    }
    //拼接合并ROI图像
    cv::Mat combined;

    // 方式 A：直接传入数组（最简洁）
    cv::Mat matrices[] = { img(regions[0]).clone(), img(regions[1]).clone(), img(regions[2]).clone(), img(regions[3]).clone() };
    cv::vconcat(matrices, 4, combined);
    //ExtractEdge(combined, r4, 0, vec_outEdgePs2);

    cv::Point2f inCenterX, outCenterX;
    CalculateCenter(vec_inEdgePs1, vec_inEdgePs2, 0, inCenterX);
    CalculateCenter(vec_outEdgePs1, vec_outEdgePs2, 0, outCenterX);
    //double distanceX = (inCenterX.x - outCenterX.x) * 0.060;
    double distanceX =std::abs(inCenterX.x - outCenterX.x);
    cout << " X1 is:" << inCenterX.x << " X2 is:" << outCenterX.x << " Distance X is:" << distanceX << endl;
    return distanceX;
}
double NccMatch::ExOntoTest1(cv::Mat img, std::vector<std::vector<double>>& vecMagn, std::vector<cv::Rect>& regions)
{
    std::vector<cv::Point2f> vec_inEdgePs1, vec_inEdgePs2, vec_outEdgePs1, vec_outEdgePs2;
    //cv::Point2f center = findMinRectTopLeft(img);

    std::vector<double> magnitude1;
    //cv::Rect r2(center.x+65, 776, 30, 80);
    ExtractEdge2(img, regions[0], 0, vec_inEdgePs1, magnitude1);
    //cv::Rect r3(center.x+233, 776, 30, 80);
    std::vector<double> magnitude2;
    ExtractEdge2(img, regions[1], 0, vec_inEdgePs2, magnitude2);


    ////cv::Rect r1(center.x-20, 760, 30, 80);
    //std::vector<double> magnitude3;
    //ExtractEdge2(img, regions[2], 0, vec_outEdgePs1, magnitude3);
    ////cv::Rect r4(center.x+317, 760, 30, 80);
    //std::vector<double> magnitude4;
    //ExtractEdge2(img, regions[3], 0, vec_outEdgePs2, magnitude4);
    //vecMagn.push_back(magnitude1);
    //vecMagn.push_back(magnitude2);
    //vecMagn.push_back(magnitude3);
    //vecMagn.push_back(magnitude4);

    //cv::Mat maImg(4, 30, CV_64F);
    //double* row_ptr = maImg.ptr<double>(0);
    //for (int j = 0; j < maImg.cols; ++j) {
    //    row_ptr[j] = magnitude1[j];
    //}
    //row_ptr = maImg.ptr<double>(1);
    //for (int j = 0; j < maImg.cols; ++j) {
    //    row_ptr[j] = magnitude2[j];
    //}
    //row_ptr = maImg.ptr<double>(2);
    //for (int j = 0; j < maImg.cols; ++j) {
    //    row_ptr[j] = magnitude3[j];
    //}
    //row_ptr = maImg.ptr<double>(3);
    //for (int j = 0; j < maImg.cols; ++j) {
    //    row_ptr[j] = magnitude4[j];
    //}
    ////拼接合并ROI图像
    //cv::Mat combined;

    //// 方式 A：直接传入数组（最简洁）
    //cv::Mat matrices[] = { img(regions[0]).clone(), img(regions[1]).clone(), img(regions[2]).clone(), img(regions[3]).clone() };
    //cv::vconcat(matrices, 4, combined);
    ////ExtractEdge(combined, r4, 0, vec_outEdgePs2);

    cv::Point2f inCenterX, outCenterX;
    CalculateCenter(vec_inEdgePs1, vec_inEdgePs2, 0, inCenterX);
    CalculateCenter(vec_outEdgePs1, vec_outEdgePs2, 0, outCenterX);
    //double distanceX = (inCenterX.x - outCenterX.x) * 0.060;
    double distanceX = std::abs(vec_inEdgePs1[0].x + vec_inEdgePs2[0].x)/2;
    //cout << " X1 is:" << inCenterX.x << " X2 is:" << outCenterX.x << " Distance X is:" << distanceX << endl;
    return distanceX;
}



