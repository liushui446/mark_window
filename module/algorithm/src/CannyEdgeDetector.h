#ifndef CANNY_EDGE_DETECTOR_HPP
#define CANNY_EDGE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>
class CannyEdgeDetector {
private:
    cv::Mat originalImage;     // 原始图像
    cv::Mat grayImage;         // 灰度图像
    cv::Mat blurredImage;      // 高斯模糊后图像
    cv::Mat gradientX, gradientY;  // X和Y方向梯度
    cv::Mat gradientMagnitude;     // 梯度幅值
    cv::Mat gradientDirection;     // 梯度方向
    cv::Mat edges;                 // 最终边缘图像

    // 高斯核生成
    cv::Mat createGaussianKernel(int size, double sigma);

    // 高斯模糊
    void gaussianBlur(int kernelSize, double sigma);

    // 计算梯度
    void computeGradients();

    // 非极大值抑制
    void nonMaximumSuppression();

    // 双阈值检测
    void doubleThreshold(double lowThreshold, double highThreshold);

public:
    // 构造函数
    CannyEdgeDetector(const cv::Mat& image);

    // 主边缘检测方法
    cv::Mat detectEdges(int kernelSize = 3,
        double sigma = 1.0,
        double lowThreshold = 30,
        double highThreshold = 50);

    // 保存边缘图像
    void saveEdges(const std::string& filename);

    // 显示边缘图像
    void displayEdges(const std::string& windowName = "Edges");

    // 获取原始图像
    cv::Mat getOriginalImage() const { return originalImage; }

    // 获取边缘图像
    cv::Mat getEdges() const { return edges; }
};

#endif // CANNY_EDGE_DETECTOR_HPP#pragma once
