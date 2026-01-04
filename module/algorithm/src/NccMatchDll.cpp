#define NCC_MATCH_DLL_EXPORTS
#include <algorithm/NccMatchDll.h>
#include "NccMatch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core/ocl.hpp>
#include "MarkTeach.h"
// 静态全局变量存储参数和匹配器实例
static double g_cannyThresh1 = 170;
static double g_cannyThresh2 = 200;
static int g_cannyApertureSize = 3;
static bool g_cannyL2gradient = true;
static double g_contourAreaThresh = 8.0;
static std::unique_ptr<NccMatch> g_matcher;


// 生成模板并保存边缘点（新增 ROI 参数支持）
NCC_MATCH_API bool NCC_CreateTemplate(
    const char* templateImagePath,
    const char* edgeXmlPath,
    const NccRect* roi,  // 新增：ROI 区域，传 NULL 表示使用全图
    int angleStep,
    int minAngle,
    int maxAngle
    
) {

    // 1. 基础参数校验（路径非空）
    if (!templateImagePath || !edgeXmlPath) {
        std::cerr << "错误：输入路径为空" << std::endl;
        return false;
    }

    try {
        // 2. 创建 NCC 匹配实例
        NccMatch matcher(angleStep, minAngle, maxAngle);

        //// 3. 设置预设参数（Canny、轮廓面积阈值）
        //matcher.setCannyParams(g_cannyThresh1, g_cannyThresh2,
        //    g_cannyApertureSize, g_cannyL2gradient);
        //matcher.setContourAreaThreshold(g_contourAreaThresh);

        // 4. 读取模板图像（灰度图）
        cv::Mat templateImage = cv::imread(templateImagePath, cv::IMREAD_GRAYSCALE);
        if (templateImage.empty()) {
            return false;
        }
        
        //////////////
        ////对比opencl加速方法
        //// 检测参数
        //double threshold1 = 50;
        //double threshold2 = 150;

        //// 结果图像
        //cv::Mat ocl_result, cpu_result;
        //cv::Mat src = cv::imread(templateImagePath, cv::IMREAD_GRAYSCALE);
        //// 运行带OpenCL加速的Canny检测
        //double ocl_time = matcher.cannyWithOpenCL(src, ocl_result, threshold1, threshold2);

        //// 运行不带OpenCL加速的Canny检测
        //double cpu_time = matcher.cannyWithoutOpenCL(src, cpu_result, threshold1, threshold2);

        //// 输出时间对比
        //std::cout << "\nPerformance Comparison:" << std::endl;
        //std::cout << "OpenCL accelerated Canny: " << ocl_time << " ms" << std::endl;
        //std::cout << "CPU only Canny: " << cpu_time << " ms" << std::endl;

        //// 计算加速比
        //if (cpu_time > 0 && ocl_time > 0) {
        //    double speedup = cpu_time / ocl_time;
        //    std::cout << "Speedup factor: " << speedup << "x" << std::endl;
        //}
        //std::cout << cv::getBuildInformation() << std::endl;

        //// 结果图像
        //cv::Mat result_ipp, result_no_ipp;

        //// 运行带IPP加速的处理（如果可用）
        //double time_ipp = matcher.processWithPossibleIPP(src, result_ipp);

        //// 运行不带IPP加速的处理
        //double time_no_ipp = matcher.processWithoutIPP(src, result_no_ipp);

        //// 输出时间对比
        //std::cout << "\n性能对比:" << std::endl;
        //std::cout << "可能启用IPP的处理时间: " << time_ipp << " 毫秒" << std::endl;
        //std::cout << "禁用优化的处理时间: " << time_no_ipp << " 毫秒" << std::endl;

        //if (time_no_ipp > 0 && time_ipp > 0) {
        //    double speedup = time_no_ipp / time_ipp;
        //    std::cout << "加速比: " << speedup << "x" << std::endl;
        //}
        // ////////////
        // 5. 处理 ROI 区域（核心逻辑）
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
            int validWidth = std::min(roi->width, templateImage.cols - validX);
            int validHeight = std::min(roi->height, templateImage.rows - validY);

            // 5.3 校验调整后的 ROI 是否有效（避免尺寸为零）
            if (validWidth <= 0 || validHeight <= 0) {
                std::cerr << "错误：ROI 超出图像范围或调整后尺寸为零" << std::endl;
                std::cerr << "图像尺寸: " << templateImage.cols << "x" << templateImage.rows << std::endl;
                std::cerr << "输入 ROI: (" << roi->x << "," << roi->y << ") " << roi->width << "x" << roi->height << std::endl;
                return false;
            }

            // 5.4 提取 ROI 区域图像
            roiImage = templateImage(cv::Rect(validX, validY, validWidth, validHeight)).clone();
            // 记录 ROI 偏移量（后续将边缘点坐标转换回原始图像坐标）
            roiOffset = cv::Point2i(validX, validY);
                   }
        else {
            // 5.5 无 ROI 时使用全图
            roiImage = templateImage.clone();
        }
        //尺寸参数提取
        //vector<RectangleInfo> crossMarkResults; // 存储检测结果的容器
        //int minFeatureSize = 20;                // 最小特征尺寸（过滤噪声，根据实际十字大小调整）

        //// -------------------------- 步骤2：调用十字标记检测函数 --------------------------
        //bool isDetectSuccess = detectFIF(roiImage, crossMarkResults, minFeatureSize);
      ////////尺寸参数提取分节线
        // 6. 保存边缘点到 XML 并生成模板
        if (!matcher.generateTemplateAndSaveEdgePoints(roiImage, edgeXmlPath)) {
            return false;
        }

        // 7. 输出成功信息
        std::cout << "边缘点已保存到: " << edgeXmlPath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        // 异常捕获（避免 DLL 崩溃）
        std::cerr << "生成模板时发生错误: " << e.what() << std::endl;
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
        
        /*if (!g_matcher->matchUsingEdgePointsFromXml(testImage, edgeXmlPath, bestLoc, bestAngle, bestScore)) {
            std::cerr << "匹配失败" << std::endl;
            return false;
        }*/
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

// 设置Canny边缘检测参数
NCC_MATCH_API void NCC_SetCannyParams(
    double thresh1,
    double thresh2,
    int apertureSize,
    bool L2gradient) {
    
    g_cannyThresh1 = thresh1;
    g_cannyThresh2 = thresh2;
    
    // 验证孔径大小
    if (apertureSize == 3 || apertureSize == 5 || apertureSize == 7) {
        g_cannyApertureSize = apertureSize;
    }
    
    g_cannyL2gradient = L2gradient;
}

// 设置轮廓面积阈值
NCC_MATCH_API void NCC_SetContourAreaThreshold(double threshold) {
    if (threshold >= 0) {
        g_contourAreaThresh = threshold;
    }
}
