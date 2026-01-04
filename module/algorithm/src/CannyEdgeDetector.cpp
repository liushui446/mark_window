#include "CannyEdgeDetector.h"
#include <iostream>
#include <cmath>

// 构造函数
CannyEdgeDetector::CannyEdgeDetector(const cv::Mat& image) : originalImage(image) {
    // 转换为灰度图
    //cv::cvtColor(originalImage, grayImage, cv::COLOR_BGR2GRAY);
    if (originalImage.channels() > 1) {
        cv::cvtColor(originalImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = originalImage.clone();
    }
}

// 高斯核生成
cv::Mat CannyEdgeDetector::createGaussianKernel(int size, double sigma) {
    cv::Mat kernel(size, size, CV_64F);
    double sum = 0.0;
    int center = size / 2;

    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            double xDist = x - center;
            double yDist = y - center;

            kernel.at<double>(x, y) = std::exp(-(xDist * xDist + yDist * yDist) / (2 * sigma * sigma));
            sum += kernel.at<double>(x, y);
        }
    }

    // 归一化
    kernel /= sum;
    return kernel;
}

// 高斯模糊
void CannyEdgeDetector::gaussianBlur(int kernelSize, double sigma) {
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(kernelSize, kernelSize), sigma);
}

// 计算梯度
void CannyEdgeDetector::computeGradients() {
    // Sobel算子计算梯度
    cv::Sobel(blurredImage, gradientX, CV_64F, 1, 0, 3);
    cv::Sobel(blurredImage, gradientY, CV_64F, 0, 1, 3);

    // 计算梯度幅值
    cv::magnitude(gradientX, gradientY, gradientMagnitude);

    // 计算梯度方向
    cv::phase(gradientX, gradientY, gradientDirection, true);
}

// 非极大值抑制
void CannyEdgeDetector::nonMaximumSuppression() {
    cv::Mat suppressedMagnitude = gradientMagnitude.clone();

    for (int y = 1; y < gradientMagnitude.rows - 1; y++) {
        for (int x = 1; x < gradientMagnitude.cols - 1; x++) {

            double magnitude = gradientMagnitude.at<double>(y, x);
            // double angle = gradientDirection.at<float>(y, x);
            double angle = gradientDirection.at<double>(y, x);
            //double angle = static_cast<double>(gradientDirection.at<float>(y, x));
            if ((angle >= 165 && angle <= 195) || (angle <= 15) || (angle >= 345))
            {
                double leftmag = gradientMagnitude.at<double>(y, x - 1);
                double rightmag = gradientMagnitude.at<double>(y, x + 1);
                if (magnitude <= leftmag || magnitude <= rightmag)
                {
                    suppressedMagnitude.at<double>(y, x) = 0;
                }
            }
            else
            {
                // 确定邻近像素
                int q = 255, r = 255;

                // 角度量化到8个方向
                if ((0 <= angle < 22.5) || (337.5 <= angle <= 360)) {
                    // 0度方向（水平方向）
                    q = gradientMagnitude.at<double>(y, x + 1);
                    r = gradientMagnitude.at<double>(y, x - 1);
                }
                else if (22.5 <= angle < 67.5) {
                    // 45度方向（右上-左下对角线）
                    q = gradientMagnitude.at<double>(y + 1, x - 1);
                    r = gradientMagnitude.at<double>(y - 1, x + 1);
                }
                else if (67.5 <= angle < 112.5) {
                    // 90度方向（垂直方向）
                    q = gradientMagnitude.at<double>(y + 1, x);
                    r = gradientMagnitude.at<double>(y - 1, x);
                }
                else if (112.5 <= angle < 157.5) {
                    // 135度方向（左上-右下对角线）
                    q = gradientMagnitude.at<double>(y - 1, x - 1);
                    r = gradientMagnitude.at<double>(y + 1, x + 1);
                }
                else if (157.5 <= angle < 202.5) {
                    // 180度方向（水平方向，反向）
                    q = gradientMagnitude.at<double>(y, x - 1);
                    r = gradientMagnitude.at<double>(y, x + 1);
                }
                else if (202.5 <= angle < 247.5) {
                    // 225度方向（右下-左上对角线）
                    q = gradientMagnitude.at<double>(y + 1, x + 1);
                    r = gradientMagnitude.at<double>(y - 1, x - 1);
                }
                else if (247.5 <= angle < 292.5) {
                    // 270度方向（垂直方向，反向）
                    q = gradientMagnitude.at<double>(y - 1, x);
                    r = gradientMagnitude.at<double>(y + 1, x);
                }
                else if (292.5 <= angle < 337.5) {
                    // 315度方向（左下-右上对角线）
                    q = gradientMagnitude.at<double>(y - 1, x + 1);
                    r = gradientMagnitude.at<double>(y + 1, x - 1);
                }

                // 非最大值抑制
                if (magnitude < q || magnitude < r) {
                    suppressedMagnitude.at<double>(y, x) = 0;
                }
            }
        }
    }

    gradientMagnitude = suppressedMagnitude;
}

// 双阈值检测
void CannyEdgeDetector::doubleThreshold(double lowThreshold, double highThreshold) {
    cv::Mat strongEdges = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);
    cv::Mat weakEdges = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);

    for (int y = 0; y < gradientMagnitude.rows; y++) {
        for (int x = 0; x < gradientMagnitude.cols; x++) {
            double magnitude = gradientMagnitude.at<double>(y, x);

            if (magnitude >= highThreshold) {
                strongEdges.at<uchar>(y, x) = 255;
            }
            else if (magnitude >= lowThreshold) {
                weakEdges.at<uchar>(y, x) = 255;
            }
        }
    }

    // 边缘追踪
    edges = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);
    for (int y = 1; y < gradientMagnitude.rows - 1; y++) {
        for (int x = 1; x < gradientMagnitude.cols - 1; x++) {
            if (strongEdges.at<uchar>(y, x) == 255) {
                edges.at<uchar>(y, x) = 255;
                // 追踪八邻域的若边缘
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (weakEdges.at<uchar>(y + dy, x + dx) == 255) {
                            edges.at<uchar>(y + dy, x + dx) = 255;
                        }
                    }
                }
            }
        }
    }
}

// 主边缘检测方法
cv::Mat CannyEdgeDetector::detectEdges(int kernelSize,
    double sigma,
    double lowThreshold,
    double highThreshold) {
    try {
        // 高斯模糊
        gaussianBlur(kernelSize, sigma);

        // 计算梯度
        computeGradients();

        // 非极大值抑制
        nonMaximumSuppression();

        // 双阈值检测
        doubleThreshold(lowThreshold, highThreshold);

        return edges;
    }
    catch (const cv::Exception& e) {
        std::cerr << "边缘检测错误: " << e.what() << std::endl;
        return cv::Mat();
    }
}

// 保存边缘图像
void CannyEdgeDetector::saveEdges(const std::string& filename) {
    if (!edges.empty()) {
        cv::imwrite(filename, edges);
    }
}

// 显示边缘图像
void CannyEdgeDetector::displayEdges(const std::string& windowName) {
    if (!edges.empty()) {
        cv::imshow(windowName, edges);
        cv::waitKey(0);
    }
}