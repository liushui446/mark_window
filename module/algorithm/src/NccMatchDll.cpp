#define NCC_MATCH_DLL_EXPORTS
#include <algorithm/NccMatchDll.h>
#include "NccMatch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core/ocl.hpp>
#include "MarkTeach.h"
#include <fstream>
// 静态全局变量存储参数和匹配器实例
static double g_cannyThresh1 = 170;
static double g_cannyThresh2 = 200;
static int g_cannyApertureSize = 3;
static bool g_cannyL2gradient = true;
static double g_contourAreaThresh = 8.0;
static std::unique_ptr<NccMatch> g_matcher;


NCC_MATCH_API bool NCC_CreateTemplate(
    const char* templateImagePath,
    const char* edgeXmlPath,
    const NccRect* roi,
    int angleStep,
    int minAngle,
    int maxAngle)
{
    if (!templateImagePath || !edgeXmlPath) {
        std::cerr << "错误：输入路径为空" << std::endl;
        return false;
    }

    try {
        // 1) 读图（灰度）
        cv::Mat templateImage = cv::imread(templateImagePath, cv::IMREAD_GRAYSCALE);
        if (templateImage.empty()) {
            std::cerr << "错误：无法读取图像 " << templateImagePath << std::endl;
            return false;
        }

        // 2) ROI 处理：用 view，不 clone（下游不修改输入）
        cv::Mat roiImage;
        if (roi != nullptr) {
            if (roi->width <= 0 || roi->height <= 0) {
                std::cerr << "错误：无效的 ROI 尺寸" << std::endl;
                return false;
            }
            const int x0 = std::max(0, roi->x);
            const int y0 = std::max(0, roi->y);
            const int w = std::min(roi->width, templateImage.cols - x0);
            const int h = std::min(roi->height, templateImage.rows - y0);
            if (w <= 0 || h <= 0) {
                std::cerr << "错误：ROI 超出图像范围" << std::endl;
                return false;
            }
            roiImage = templateImage(cv::Rect(x0, y0, w, h));   // 不 clone
        }
        else {
            roiImage = templateImage;  // 共享数据，无拷贝
        }

        // 3) 生成模板
        // 创建全局 matcher，所有模板数据永久保存
        g_matcher = std::make_unique<NccMatch>(angleStep, minAngle, maxAngle);
        return g_matcher->generateTemplateAndSaveEdgePoints(roiImage, edgeXmlPath);
        /*NccMatch matcher(angleStep, minAngle, maxAngle);
        return matcher.generateTemplateAndSaveEdgePoints(roiImage, edgeXmlPath);*/
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV 异常: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "生成模板异常: " << e.what() << std::endl;
        return false;
    }
}

// 使用保存的边缘点执行匹配
NCC_MATCH_API bool NCC_PerformMatching(
    const char* testImagePath,
    const char* edgeXmlPath,
    const NccRect* roi,
    NccMatchResult* result) {
    auto start = std::chrono::high_resolution_clock::now();
    if (!testImagePath || !edgeXmlPath || !result) {
        std::cerr << "错误：输入参数为空" << std::endl;
        return false;
    }

    // 初始化结果
    result->success = false;
    result->x = 0;
    result->y = 0;
    result->angle = 0;
    result->similarity = 0;

    try {
        // 创建或重置匹配器实例
        if (!g_matcher) {
            g_matcher = std::make_unique<NccMatch>();
        }
        
        //// 设置参数
        //g_matcher->setCannyParams(g_cannyThresh1, g_cannyThresh2, 
        //                         g_cannyApertureSize, g_cannyL2gradient);
        //g_matcher->setContourAreaThreshold(g_contourAreaThresh);
        
        // 读取待检测图像（灰度图）
        cv::Mat testImage = cv::imread(testImagePath, cv::IMREAD_GRAYSCALE);
        if (testImage.empty()) {
            return false;
        }
        
        // 从XML加载边缘点并执行匹配
        cv::Mat roiImage;          // 最终用于边缘检测的图像（ROI 裁剪后或全图）
        cv::Point2i roiOffset(0, 0); // ROI 相对于原始图像的偏移量（用于坐标转换）

        if (roi != nullptr) {
            // 5.1 校验 ROI 参数有效性（坐标非负、尺寸为正）
            if (roi->x < 0 || roi->y < 0 || roi->width <= 0 || roi->height <= 0) {
                std::cerr << "错误：无效的 ROI 参数（坐标/尺寸不能为负或零）" << std::endl;
                return false;
            }

            // 5.2 确保 ROI 不超出原始图像范围（自动裁剪超出部分）
            int validX = std::max(0, roi->x);
            int validY = std::max(0, roi->y);
            int validWidth = std::min(roi->width, testImage.cols - validX);
            int validHeight = std::min(roi->height, testImage.rows - validY);

            // 5.3 校验调整后的 ROI 是否有效（避免尺寸为零）
            if (validWidth <= 0 || validHeight <= 0) {
                return false;
            }

            // 5.4 提取 ROI 区域图像
            roiImage = testImage(cv::Rect(validX, validY, validWidth, validHeight)).clone();
            // 记录 ROI 偏移量（后续将边缘点坐标转换回原始图像坐标）
            roiOffset = cv::Point2i(validX, validY);
        }
        else {
            // 5.5 无 ROI 时使用全图
            roiImage = testImage.clone();
        }
        cv::Point2f bestLoc;
        float bestAngle;  // 弧度
        double bestScore;
        bool matchSuccess1 = g_matcher->runFullMatchingFromPath(roiImage, edgeXmlPath, bestLoc, bestAngle, bestScore);
        if (!matchSuccess1) {
            std::cerr << "NCC_PerformMatching: runFullMatchingFromPath failed" << std::endl;
            return false;
        }
        /*if (!g_matcher->matchUsingEdgePointsFromXml(testImage, edgeXmlPath, bestLoc, bestAngle, bestScore)) {
            std::cerr << "匹配失败" << std::endl;
            return false;
        }*/
        std::cerr << "NCC match result: loc=(" << bestLoc.x << "," << bestLoc.y << ") angle=" << bestAngle << " score=" << bestScore << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        // 填充结果
        if (roi != nullptr)
        {
            result->success = true;
            result->x = bestLoc.x + roi->x;
            result->y = bestLoc.y + roi->y;
            result->angle = bestAngle;  // 转换为度
            result->similarity = static_cast<float>(bestScore);
            result->time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }
        else
        {
            result->success = true;
            result->x = bestLoc.x;
            result->y = bestLoc.y;
            result->angle = bestAngle;  // 转换为度
            result->similarity = static_cast<float>(bestScore);
            result->time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }
        
        
        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

NCC_MATCH_API bool Region_PerformMatching(
    const char* testImagePath,
    const char* edgeXmlPath,
    const NccRect* roi,
    NccMatchResult* result,
    int markType) {  // 新增markType参数

    auto start = std::chrono::high_resolution_clock::now();
    if (!testImagePath || !edgeXmlPath || !result) {
        return false;
    }

    // 初始化结果
    result->success = false;
    result->x = 0;
    result->y = 0;
    result->angle = 0;
    result->similarity = 0;

    try {
        // 创建或重置匹配器实例
        if (!g_matcher) {
            g_matcher = std::make_unique<NccMatch>();
        }
        cv::Mat testImage = cv::imread(testImagePath, cv::IMREAD_GRAYSCALE);
        if (testImage.empty()) {
            return false;
        }

        // 从XML加载边缘点并执行匹配
        cv::Mat roiImage;          // 最终用于边缘检测的图像（ROI 裁剪后或全图）
        cv::Point2i roiOffset(0, 0); // ROI 相对于原始图像的偏移量（用于坐标转换）

        if (roi != nullptr) {
            // 5.1 校验 ROI 参数有效性（坐标非负、尺寸为正）
            if (roi->x < 0 || roi->y < 0 || roi->width <= 0 || roi->height <= 0) {
                return false;
            }

            // 5.2 确保 ROI 不超出原始图像范围（自动裁剪超出部分）
            int validX = std::max(0, roi->x);
            int validY = std::max(0, roi->y);
            int validWidth = std::min(roi->width, testImage.cols - validX);
            int validHeight = std::min(roi->height, testImage.rows - validY);

            // 5.3 校验调整后的 ROI 是否有效（避免尺寸为零）
            if (validWidth <= 0 || validHeight <= 0) {
                return false;
            }
            roiImage = testImage(cv::Rect(validX, validY, validWidth, validHeight)).clone();
            roiOffset = cv::Point2i(validX, validY);
        }
        else {
            // 5.5 无 ROI 时使用全图
            roiImage = testImage.clone();
        }
        cv::Point2f bestLoc;
        float bestAngle;  // 弧度
        double bestScore;

        // 修改：传递markType参数到Region_test_subpix
        bool matchSuccess1 = g_matcher->Region_test(roiImage, edgeXmlPath, bestLoc, bestAngle, bestScore, markType);

        std::cerr << "NCC match result: loc=(" << bestLoc.x << "," << bestLoc.y << ") angle=" << bestAngle << " score=" << bestScore << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        // 填充结果
        if (roi != nullptr)
        {
            result->success = true;
            result->x = bestLoc.x + roi->x;
            result->y = bestLoc.y + roi->y;
            result->angle = bestAngle;  // 转换为度
            result->similarity = static_cast<float>(bestScore);
            result->time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }
        else
        {
            result->success = true;
            result->x = bestLoc.x;
            result->y = bestLoc.y;
            result->angle = bestAngle;  // 转换为度
            result->similarity = static_cast<float>(bestScore);
            result->time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }
        result->x = bestLoc.x;
        result->y = bestLoc.y;

        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

