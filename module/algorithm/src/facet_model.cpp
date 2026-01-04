#include "facet_model.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include "CannyEdgeDetector.h"
#include "CannyEdge.h"
#include "sim_template.h"
FacetEdgeDetector::FacetEdgeDetector(int windowSize) {
    this->windowSize = windowSize;
    initKernels();
}
// 预计算模板梯度信息
struct PrecomputedData1 {
    cv::Mat gradX, gradY; // 存储梯度信息
    cv::Size templateSize;
};

PrecomputedData1 precomputeTemplateData1(const cv::Mat& gradX1, const cv::Mat& gradY1) {
    PrecomputedData1 data;
    data.gradX = gradX1.clone();
    data.gradY = gradY1.clone();
    data.templateSize = gradX1.size();
    return data;
}
// 初始化facet模型的卷积核
void FacetEdgeDetector::initKernels() {
    kernels.resize(10);

    // 计算窗口半径
    int radius = windowSize / 2;

    // 为每个基函数创建卷积核
    for (int i = 0; i < 10; i++) {
        kernels[i] = cv::Mat::zeros(windowSize, windowSize, CV_32F);
    }

    // 计算分母部分 Σ Pi²(r,c)
    std::vector<float> denominators(10, 0.0f);

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            // 计算10个基函数在(x,y)处的值
            std::vector<float> basis = computeBasisFunctions(x, y);

            // 更新分母
            for (int i = 0; i < 10; i++) {
                denominators[i] += basis[i] * basis[i];
            }

            // 设置卷积核的值
            for (int i = 0; i < 10; i++) {
                kernels[i].at<float>(y + radius, x + radius) = basis[i];
            }
        }
    }

    // 正规化卷积核
    for (int i = 0; i < 10; i++) {
        kernels[i] /= denominators[i];
    }
}

// 计算基函数在点(x,y)的值
std::vector<float> FacetEdgeDetector::computeBasisFunctions(int x, int y) {
    std::vector<float> basis(10);
    float x2 = x * x;
    float y2 = y * y;

    // {1, x, y, x² - 2, xy, y² - 2, x³ - x·17.0/5.0, (x² - 2)y, x(y² - 2), y³ - y·17.0/5.0}
    basis[0] = 1.0f;
    basis[1] = x;
    basis[2] = y;
    basis[3] = x2 - 2.0f;
    basis[4] = x * y;
    basis[5] = y2 - 2.0f;
    basis[6] = x * x2 - x * 17.0f / 5.0f;
    basis[7] = (x2 - 2.0f) * y;
    basis[8] = x * (y2 - 2.0f);
    basis[9] = y * y2 - y * 17.0f / 5.0f;

    return basis;
}

// 计算极坐标下的亚像素位置
cv::Point2f FacetEdgeDetector::computeSubpixelInPolar(
    float x, float y,
    float k2, float k3, float k4, float k5, float k6,
    float k7, float k8, float k9, float k10
) {
    // 计算梯度幅值
    float magnitude = std::sqrt(k2 * k2 + k3 * k3);

    // 如果梯度为零，返回原始位置
    if (magnitude < 1e-6) {
        return cv::Point2f(x, y);
    }

    // 计算梯度单位向量(余弦和正弦值)
    float cosAlpha = k2 / magnitude;
    float sinAlpha = k3 / magnitude;

    // 计算ρ (沿梯度方向的位移)
    float numerator = -(k4 * cosAlpha * cosAlpha + k5 * cosAlpha * sinAlpha + k6 * sinAlpha * sinAlpha);
    float denominator = 3 * (k7 * std::pow(cosAlpha, 3) +
        k8 * std::pow(cosAlpha, 2) * sinAlpha +
        k9 * cosAlpha * std::pow(sinAlpha, 2) +
        k10 * std::pow(sinAlpha, 3));

    // 计算ρ，并处理退化情况
    float rho = 0.0f;
    if (std::abs(denominator) > 1e-6) {
        rho = numerator / denominator;
    }
    else {
        // 退化为二阶导数方法
        float gxx = 2 * k4;
        float gxy = k5;
        float gyy = 2 * k6;

        // 沿梯度方向的二阶导数
        float dirxx = gxx * cosAlpha * cosAlpha + 2 * gxy * cosAlpha * sinAlpha + gyy * sinAlpha * sinAlpha;

        if (std::abs(dirxx) > 1e-6) {
            rho = -magnitude / dirxx;
        }
    }

    // 限制位移在合理范围内（±0.5像素）
    rho = std::max(-0.5f, std::min(0.5f, rho));

    // 计算最终亚像素位置
    return cv::Point2f(x + rho * cosAlpha, y + rho * sinAlpha);
}

// 使用facet模型进行亚像素边缘检测
cv::Mat FacetEdgeDetector::detectEdges(const cv::Mat& image, float threshold) {
    // 转换为灰度图像
    cv::Mat grayImage;
    if (image.channels() > 1) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = image.clone();
    }

    //// 转换为浮点型
    //grayImage.convertTo(grayImage, CV_32F);

    // 使用卷积核计算10个系数
    //cv::GaussianBlur(grayImage, grayImage,cv::Size(3,3),3, 3);
    std::vector<cv::Mat> coefficients(10);
    for (int i = 0; i < 10; i++) {
        cv::filter2D(grayImage, coefficients[i], CV_32F, kernels[i]);
    }

    // 创建结果图像
    cv::Mat edgeImage = cv::Mat::zeros(grayImage.size(), CV_32F);
    cv::Mat edgeAngle = cv::Mat::zeros(grayImage.size(), CV_32F);
    cv::Mat subpixelX = cv::Mat::zeros(grayImage.size(), CV_32F);
    cv::Mat subpixelY = cv::Mat::zeros(grayImage.size(), CV_32F);

    // 从系数计算梯度
    cv::Mat gradX = coefficients[1].clone();  // a_x 对应于 x 基函数的系数
    cv::Mat gradY = coefficients[2].clone();  // a_y 对应于 y 基函数的系数

    // 从系数计算二阶导数
    cv::Mat fxx = 2 * coefficients[3].clone();  // a_xx 对应于 x² - 2 基函数的系数的2倍
    cv::Mat fxy = coefficients[4].clone();      // a_xy 对应于 xy 基函数的系数
    cv::Mat fyy = 2 * coefficients[5].clone();  // a_yy 对应于 y² - 2 基函数的系数的2倍

    // 从系数计算三阶导数
    cv::Mat fxxx = 6 * coefficients[6].clone();  // a_xxx 对应于 x³ - x·17.0/5.0 基函数的系数
    cv::Mat fxxy = 2 * coefficients[7].clone();  // a_xxy 对应于 (x² - 2)y 基函数的系数
    cv::Mat fxyy = 2 * coefficients[8].clone();  // a_xyy 对应于 x(y² - 2) 基函数的系数
    cv::Mat fyyy = 6 * coefficients[9].clone();  // a_yyy 对应于 y³ - y·17.0/5.0 基函数的系数
    //
    //// 创建边缘检测器
    //CannyEdgeDetector detector(image);

    //// 执行边缘检测
    //cv::Mat edges = detector.detectEdges(
    //    5,     // 高斯核大小
    //    2,   // 高斯sigma
    //    25,    // 低阈值 
    //    45     // 高阈值
    //);
    //
    std::vector<cv::Point> edgePoints;
    cv::Mat edge = CannyEdge::detectEdges(image, edgePoints);

    
    for (int y = 0; y < edge.rows; y++) {
        for (int x = 0; x < edge.cols; x++) {
            if (edge.at<uchar>(y, x) > 0) {
                edgePoints.push_back(cv::Point(x, y));
            }
        }
    }
    ////存储像素级点
    //// 将边缘点保存到txt文件
    //std::ofstream outFile("edge_points.txt");
    //for (const auto& point : edgePoints) {
    //    outFile << point.x << " " << point.y << std::endl;
    //}
    //outFile.close();
    // 1. 打开并读取 edges.txt 文件
    //std::ifstream file("D:/mark_window/module/algorithm/src/edge_points.txt");

    ////std::vector<cv::Point> edgePoints; // 用来存储边缘点
    //edgePoints.clear();
    //int x, y;
    //// 2. 读取每一行，直到文件结束
    //while (file >> x >> y) {
    //    // 将读取到的坐标存储到 vector 中
    //    edgePoints.push_back(cv::Point(x, y));
    //}

    //file.close(); // 关闭文件
    std::ifstream file("edge_points.txt");  // 使用当前文件夹的相对路径

    edgePoints.clear();
    int x, y;
    // 2. 读取每一行，直到文件结束
    while (file >> x >> y) {
        // 将读取到的坐标存储到 vector 中
        edgePoints.push_back(cv::Point(x, y));
    }

    // 建议添加文件打开检查
    if (!file.is_open()) {
        std::cerr << "无法打开edge_points.txt文件，请检查是否存在" << std::endl;
        // 错误处理逻辑
    }

    file.close(); // 关闭文件
    std::vector<cv::Point2f> subedgePoints;
    for (int i = 0; i < edgePoints.size(); i++)
    {
        int x = edgePoints.at(i).x;
        int y = edgePoints.at(i).y;
        // 获取所有系数值 (k1到k10)
        float k1 = coefficients[0].at<float>(y, x) - 2 * coefficients[3].at<float>(y, x) - 2 * coefficients[5].at<float>(y, x);  // 常数项
        float k2 = coefficients[1].at<float>(y, x) - 2 * coefficients[8].at<float>(y, x) - coefficients[6].at<float>(y, x) * 17 / 5;  // x系数
        float k3 = coefficients[2].at<float>(y, x) - 2 * coefficients[7].at<float>(y, x) - coefficients[9].at<float>(y, x) * 17 / 5;  // y系数
        float k4 = coefficients[3].at<float>(y, x);  // x²-2系数
        float k5 = coefficients[4].at<float>(y, x);  // xy系数
        float k6 = coefficients[5].at<float>(y, x);  // y²-2系数
        float k7 = coefficients[6].at<float>(y, x);  // x³-x·17/5系数
        float k8 = coefficients[7].at<float>(y, x);  // (x²-2)y系数
        float k9 = coefficients[8].at<float>(y, x);  // x(y²-2)系数
        float k10 = coefficients[9].at<float>(y, x); // y³-y·17/5系数
        // 使用极坐标方法计算亚像素位置
        // 计算梯度幅值和方向
        float magnitude = std::sqrt(k2 * k2 + k3 * k3);
        float angle = std::atan2(k3, k2);
        // 1. 计算α (梯度角度)
        float cosAlpha = k2 / magnitude;
        float sinAlpha = k3 / magnitude;

        // 2. 计算ρ (沿梯度方向的位移)
        float numerator = -(k4 * cosAlpha * cosAlpha + k5 * cosAlpha * sinAlpha + k6 * sinAlpha * sinAlpha);
        float denominator = 3 * (k7 * std::pow(cosAlpha, 3) +
            k8 * std::pow(cosAlpha, 2) * sinAlpha +
            k9 * cosAlpha * std::pow(sinAlpha, 2) +
            k10 * std::pow(sinAlpha, 3));

        // 如果分母接近零，使用替代方法
        float rho = 0.0f;
        if (std::abs(denominator) > 1e-6) {
            rho = numerator / denominator;

            // 限制位移在合理范围内（±0.5像素）
            if (abs(rho) > 0.5)
            {
                rho = std::max(-0.5f, std::min(0.5f, rho));
                //rho = 0;
            }
            
        }
        else {
            // 退化为二阶导数方法
            float gxx = 2 * k4;
            float gxy = k5;
            float gyy = 2 * k6;

            // 沿梯度方向的二阶导数
            float dirxx = gxx * cosAlpha * cosAlpha + 2 * gxy * cosAlpha * sinAlpha + gyy * sinAlpha * sinAlpha;

            if (std::abs(dirxx) > 1e-6) {
                rho = -magnitude / dirxx;
                rho = std::max(-0.5f, std::min(0.5f, rho));
            }
        }

        // 3. 计算亚像素位置
        float subpixelXVal = x + rho * cosAlpha;
        float subpixelYVal = y + rho * sinAlpha;
        subedgePoints.push_back(cv::Point2f(subpixelXVal, subpixelYVal));
        // 存储结果
        edgeImage.at<float>(y, x) = magnitude;
        edgeAngle.at<float>(y, x) = angle;
        subpixelX.at<float>(y, x) = subpixelXVal;
        subpixelY.at<float>(y, x) = subpixelYVal;
    }
    /*std::string outPath = "D:/mark_window/module/algorithm/src/sub_edge_points.txt";
    std::ofstream outFile1(outPath);
    for (const auto& point : subedgePoints) {
        outFile1 << point.x << " " << point.y << std::endl;
    }
    outFile1.close();*/
    std::string outPath = "sub_edge_points.txt";
    std::ofstream outFile1(outPath);

    for (const auto& point : subedgePoints) {
        outFile1 << point.x << " " << point.y << std::endl;
    }
    outFile1.close();
    // 计算亚像素边缘
    //for (int y = 1; y < grayImage.rows - 1; y++) {
    //    for (int x = 1; x < grayImage.cols - 1; x++) {
    //        // 获取所有系数值 (k1到k10)
    //        float k1 = coefficients[0].at<float>(y, x);  // 常数项
    //        float k2 = coefficients[1].at<float>(y, x);  // x系数
    //        float k3 = coefficients[2].at<float>(y, x);  // y系数
    //        float k4 = coefficients[3].at<float>(y, x);  // x²-2系数
    //        float k5 = coefficients[4].at<float>(y, x);  // xy系数
    //        float k6 = coefficients[5].at<float>(y, x);  // y²-2系数
    //        float k7 = coefficients[6].at<float>(y, x);  // x³-x·17/5系数
    //        float k8 = coefficients[7].at<float>(y, x);  // (x²-2)y系数
    //        float k9 = coefficients[8].at<float>(y, x);  // x(y²-2)系数
    //        float k10 = coefficients[9].at<float>(y, x); // y³-y·17/5系数

    //        // 计算梯度幅值和方向
    //        float magnitude = std::sqrt(k2 * k2 + k3 * k3);
    //        float angle = std::atan2(k3, k2);

    //        // 如果梯度幅值超过阈值，则进行亚像素分析
    //        if (magnitude > threshold) {
    //            // 使用极坐标方法计算亚像素位置

    //            // 1. 计算α (梯度角度)
    //            float cosAlpha = k2 / magnitude;
    //            float sinAlpha = k3 / magnitude;

    //            // 2. 计算ρ (沿梯度方向的位移)
    //            float numerator = -(k4 * cosAlpha * cosAlpha + k5 * cosAlpha * sinAlpha + k6 * sinAlpha * sinAlpha);
    //            float denominator = 3 * (k7 * std::pow(cosAlpha, 3) +
    //                k8 * std::pow(cosAlpha, 2) * sinAlpha +
    //                k9 * cosAlpha * std::pow(sinAlpha, 2) +
    //                k10 * std::pow(sinAlpha, 3));

    //            // 如果分母接近零，使用替代方法
    //            float rho = 0.0f;
    //            if (std::abs(denominator) > 1e-6) {
    //                rho = numerator / denominator;

    //                // 限制位移在合理范围内（±0.5像素）
    //                rho = std::max(-0.5f, std::min(0.5f, rho));
    //            }
    //            else {
    //                // 退化为二阶导数方法
    //                float gxx = 2 * k4;
    //                float gxy = k5;
    //                float gyy = 2 * k6;

    //                // 沿梯度方向的二阶导数
    //                float dirxx = gxx * cosAlpha * cosAlpha + 2 * gxy * cosAlpha * sinAlpha + gyy * sinAlpha * sinAlpha;

    //                if (std::abs(dirxx) > 1e-6) {
    //                    rho = -magnitude / dirxx;
    //                    rho = std::max(-0.5f, std::min(0.5f, rho));
    //                }
    //            }

    //            // 3. 计算亚像素位置
    //            float subpixelXVal = x + rho * cosAlpha;
    //            float subpixelYVal = y + rho * sinAlpha;

    //            // 存储结果
    //            edgeImage.at<float>(y, x) = magnitude;
    //            edgeAngle.at<float>(y, x) = angle;
    //            subpixelX.at<float>(y, x) = subpixelXVal;
    //            subpixelY.at<float>(y, x) = subpixelYVal;
    //        }
    //    }
    //}

    return edge;
}

// 获取亚像素边缘点
std::vector<cv::Point2f> FacetEdgeDetector::getSubpixelEdgePoints(const cv::Mat& image, float threshold) {
    // 转换为灰度图像
    cv::Mat grayImage;
    if (image.channels() > 1) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = image.clone();
    }

    // 转换为浮点型
    grayImage.convertTo(grayImage, CV_32F);

    // 使用卷积核计算10个系数
    std::vector<cv::Mat> coefficients(10);
    for (int i = 0; i < 10; i++) {
        cv::filter2D(grayImage, coefficients[i], CV_32F, kernels[i]);
    }

    // 存储亚像素边缘点
    std::vector<cv::Point2f> edgePoints;

    // 从系数计算梯度
    cv::Mat gradX = coefficients[1].clone();
    cv::Mat gradY = coefficients[2].clone();

    // 从系数计算二阶导数
    cv::Mat fxx = 2 * coefficients[3].clone();
    cv::Mat fxy = coefficients[4].clone();
    cv::Mat fyy = 2 * coefficients[5].clone();

    // 计算亚像素边缘
    for (int y = 1; y < grayImage.rows - 1; y++) {
        for (int x = 1; x < grayImage.cols - 1; x++) {
            // 获取所有系数值 (k1到k10)
            float k1 = coefficients[0].at<float>(y, x);  // 常数项
            float k2 = coefficients[1].at<float>(y, x);  // x系数
            float k3 = coefficients[2].at<float>(y, x);  // y系数
            float k4 = coefficients[3].at<float>(y, x);  // x²-2系数
            float k5 = coefficients[4].at<float>(y, x);  // xy系数
            float k6 = coefficients[5].at<float>(y, x);  // y²-2系数
            float k7 = coefficients[6].at<float>(y, x);  // x³-x·17/5系数
            float k8 = coefficients[7].at<float>(y, x);  // (x²-2)y系数
            float k9 = coefficients[8].at<float>(y, x);  // x(y²-2)系数
            float k10 = coefficients[9].at<float>(y, x); // y³-y·17/5系数

            // 计算梯度幅值
            float magnitude = std::sqrt(k2 * k2 + k3 * k3);

            // 如果梯度幅值超过阈值，则进行亚像素分析
            if (magnitude > threshold) {
                // 使用极坐标方法计算亚像素位置

                // 1. 计算α (梯度角度)
                float cosAlpha = k2 / magnitude;
                float sinAlpha = k3 / magnitude;

                // 2. 计算ρ (沿梯度方向的位移)
                float numerator = -(k4 * cosAlpha * cosAlpha + k5 * cosAlpha * sinAlpha + k6 * sinAlpha * sinAlpha);
                float denominator = 3 * (k7 * std::pow(cosAlpha, 3) +
                    k8 * std::pow(cosAlpha, 2) * sinAlpha +
                    k9 * cosAlpha * std::pow(sinAlpha, 2) +
                    k10 * std::pow(sinAlpha, 3));

                // 如果分母接近零，使用替代方法
                float rho = 0.0f;
                if (std::abs(denominator) > 1e-6) {
                    rho = numerator / denominator;

                    // 限制位移在合理范围内（±0.5像素）
                    rho = std::max(-0.5f, std::min(0.5f, rho));
                }
                else {
                    // 退化为二阶导数方法
                    float gxx = 2 * k4;
                    float gxy = k5;
                    float gyy = 2 * k6;

                    // 沿梯度方向的二阶导数
                    float dirxx = gxx * cosAlpha * cosAlpha + 2 * gxy * cosAlpha * sinAlpha + gyy * sinAlpha * sinAlpha;

                    if (std::abs(dirxx) > 1e-6) {
                        rho = -magnitude / dirxx;
                        rho = std::max(-0.5f, std::min(0.5f, rho));
                    }
                }

                // 3. 计算亚像素位置
                float subpixelX = x + rho * cosAlpha;
                float subpixelY = y + rho * sinAlpha;

                // 添加亚像素边缘点
                edgePoints.push_back(cv::Point2f(subpixelX, subpixelY));
            }
        }
    }

    return edgePoints;
}

// 可视化亚像素边缘点
cv::Mat FacetEdgeDetector::visualizeEdgePoints(const cv::Mat& image, const std::vector<cv::Point2f>& edgePoints) {
    cv::Mat result = image.clone();
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }

    // 绘制亚像素边缘点
    for (const auto& pt : edgePoints) {
        cv::circle(result, pt, 1, cv::Scalar(0, 0, 255), -1);
    }

    return result;
}

// 计算梯度向量的函数
void FacetEdgeDetector::calculateGradientVectors(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY) {
    // 转换为浮点型
    //image.convertTo(image, CV_32F);

    // 使用卷积核计算10个系数
    std::vector<cv::Mat> coefficients(10);
    for (int i = 0; i < 10; i++) {
        cv::filter2D(image, coefficients[i], CV_32F, kernels[i]);
    }
    // 保存本次的 coefficients
    allCoefficients.push_back(coefficients);
    // 获取图像尺寸
    int rows = coefficients[0].rows;
    int cols = coefficients[0].cols;

    // 初始化梯度矩阵
    gradX = cv::Mat::zeros(rows, cols, CV_32F);
    gradY = cv::Mat::zeros(rows, cols, CV_32F);

    // 根据公式计算梯度
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            // 获取系数
            float k2 = coefficients[1].at<float>(y, x) - 2 * coefficients[8].at<float>(y, x) - coefficients[6].at<float>(y, x) * 17 / 5;  // x系数
            float k3 = coefficients[2].at<float>(y, x) - 2 * coefficients[7].at<float>(y, x) - coefficients[9].at<float>(y, x) * 17 / 5;  // y系数
            float k4 = coefficients[3].at<float>(y, x);  // x²-2系数
            float k5 = coefficients[4].at<float>(y, x);  // xy系数
            float k6 = coefficients[5].at<float>(y, x);  // y²-2系数
            float k7 = coefficients[6].at<float>(y, x);  // x³-x·17/5系数
            float k8 = coefficients[7].at<float>(y, x);  // (x²-2)y系数
            float k9 = coefficients[8].at<float>(y, x);  // x(y²-2)系数
            float k10 = coefficients[9].at<float>(y, x); // y³-y·17/5系数

            // 计算 x 方向梯度: k2 + 2*k4*x + k5*y + 3*k7*x² + 2*k8*x*y + k9*y²
            gradX.at<float>(y, x) = k2 + 2 * k4 * x + k5 * y + 3 * k7 * x * x + 2 * k8 * x * y + k9 * y * y;

            // 计算 y 方向梯度: k3 + k5*x + 2*k6*y + k8*x² + 2*k9*x*y + 3*k10*y²
            gradY.at<float>(y, x) = k3 + k5 * x + 2 * k6 * y + k8 * x * x + 2 * k9 * x * y + 3 * k10 * y * y;
        }
    }
}
// 根据亚像素位置计算梯度向量的函数
void FacetEdgeDetector::calculateGradientVectors(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY, std::vector<cv::Point>& edgePoints) {
    // 转换为浮点型
    //image.convertTo(image, CV_32F);

    // 使用卷积核计算10个系数
    std::vector<cv::Mat> coefficients(10);
    for (int i = 0; i < 10; i++) {
        cv::filter2D(image, coefficients[i], CV_32F, kernels[i]);
    }
    // 保存本次的 coefficients
    //allCoefficients.push_back(coefficients);
    // 获取图像尺寸
    int rows = coefficients[0].rows;
    int cols = coefficients[0].cols;

    // 初始化梯度矩阵
    gradX = cv::Mat::zeros(rows, cols, CV_32F);
    gradY = cv::Mat::zeros(rows, cols, CV_32F);
    std::vector<cv::Point2f> subedgePoints;
    std::ifstream inFile("sub_edge_points.txt");
    float readX, readY;
    while (inFile >> readX >> readY) {
        subedgePoints.emplace_back(readX, readY);
    }
    inFile.close();
    std::ofstream outFile("gradient_vectors.txt");
    if (!outFile.is_open()) {
        std::cerr << "无法打开输出文件 gradient_vectors.txt" << std::endl;
        return;
    }
    // 根据公式计算梯度
    for (size_t i = 0; i < edgePoints.size(); ++i) {
        double x = static_cast<double>(edgePoints[i].x);
        double y = static_cast<double>(edgePoints[i].y);
        // 获取系数
        double k2 = coefficients[1].at<float>(y, x) - 2 * coefficients[8].at<float>(y, x) - coefficients[6].at<float>(y, x) * 17 / 5;  // x系数
        double k3 = coefficients[2].at<float>(y, x) - 2 * coefficients[7].at<float>(y, x) - coefficients[9].at<float>(y, x) * 17 / 5;  // y系数
        double k4 = coefficients[3].at<float>(y, x);  // x²-2系数
        double k5 = coefficients[4].at<float>(y, x);  // xy系数
        double k6 = coefficients[5].at<float>(y, x);  // y²-2系数
        double k7 = coefficients[6].at<float>(y, x);  // x³-x·17/5系数
        double k8 = coefficients[7].at<float>(y, x);  // (x²-2)y系数
        double k9 = coefficients[8].at<float>(y, x);  // x(y²-2)系数
        double k10 = coefficients[9].at<float>(y, x); // y³-y·17/5系数
        /*double newX = static_cast<double>(subedgePoints[i].x);
        double newY = static_cast<double>(subedgePoints[i].y);*/
        double newX = x;
        double newY = y;
        double gx1 = k2 + 2 * k4 * newX + k5 * newY + 3 * k7 * newX * newX + 2 * k8 * newX * newY + k9 * newY * newY;
        double gy1 = k3 + k5 * newX + 2 * k6 * newY + k8 * newX * newX + 2 * k9 * newX * newY + 3 * k10 * newY * newY;
        gradX.at<float>(y, x) = gx1;
        gradY.at<float>(y, x) = gy1;
        // 写入文件，一行两个值
        outFile << gx1 << " " << gy1 << std::endl;
    }
    outFile.close();
}
// 创建初始模板
void FacetEdgeDetector::saveGradientVectors(const cv::Mat& image, const cv::Mat& gradX, const cv::Mat& gradY) {

    // 第一步：读取sub_edge_points_cropped.txt里的点
    std::vector<cv::Point2f> subedgePoints;
    std::ifstream inFile("sub_edge_points_cropped.txt");
    float readX, readY;
    while (inFile >> readX >> readY) {
        subedgePoints.emplace_back(readX, readY);
    }
    inFile.close();
    std::ofstream outFile("gradient_vectors.txt");
    if (!outFile.is_open()) {
        std::cerr << "无法打开输出文件 gradient_vectors.txt" << std::endl;
        return;
    }

    for (const auto& pt : subedgePoints) {
        float baseX = pt.x;
        float baseY = pt.y;

        float gx1 = gradX.at<float>(baseY, baseX);  // 注意是 row = y, col = x
        float gy1 = gradY.at<float>(baseY, baseX);



        // 写入文件，一行两个值
        outFile << gx1 << " " << gy1 << std::endl;
    }

    outFile.close();
}
// 使用梯度向量进行模板匹配mark
cv::Point2f FacetEdgeDetector::gradientTemplateMatching(const cv::Mat& templateImage, const cv::Mat& targetImage, cv::Point& best_position, std::vector<cv::Point>& edgePoints, const cv::Mat& templateGradX, const cv::Mat& templateGradY) {
    //// 计算模板图像的系数和梯度
    //cv::Mat templateGradX, templateGradY;
    //calculateGradientVectors(templateImage, templateGradX, templateGradY);
   // saveGradientVectors(templateImage, templateGradX, templateGradY);
    // 计算目标图像的系数和梯度
    cv::Mat targetGradX, targetGradY;
    calculateGradientVectors(targetImage, targetGradX, targetGradY);
    // 预处理模板数据
    PrecomputedData1 templateData = precomputeTemplateData1(templateGradX, templateGradY);



    // 多尺度搜索参数
    const int SCALES[] = { 4, 2, 1 };  // 多级缩放
    const int STEPS[] = { 8, 4, 1 };  // 各级步长

    cv::Point bestPos(0, 0);
    double maxSimilarity = -1;

    //    for (int scaleIdx = 0; scaleIdx < 3; ++scaleIdx) {
    //        int scale = SCALES[scaleIdx];
    //        int step = STEPS[scaleIdx];
    //
    //        int startX = std::max(0, bestPos.x - 2 * step);
    //        int startY = std::max(0, bestPos.y - 2 * step);
    //        int endX = std::min(targetImage.cols - templateGradX.cols, bestPos.x + 2 * step);
    //        int endY = std::min(targetImage.rows - templateGradX.rows, bestPos.y + 2 * step);
    //
    //        if (scaleIdx == 0) {
    //            startX = 0;
    //            startY = 0;
    //            endX = targetImage.cols - templateGradX.cols;
    //            endY = targetImage.rows - templateGradX.rows;
    //        }
    //
    //#pragma omp parallel for collapse(2)
    //        for (int y = startY; y <= endY; y += step) {
    //            for (int x = startX; x <= endX; x += step) {
    //                cv::Mat gradXROI = targetGradX(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));
    //                cv::Mat gradYROI = targetGradY(cv::Rect(x, y, templateGradY.cols, templateGradY.rows));
    //
    //                double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
    //                    gradXROI, gradYROI, edgePoints);
    //
    //#pragma omp critical
    //                {
    //                    if (similarity > maxSimilarity) {
    //                        maxSimilarity = similarity;
    //                        bestPos = cv::Point(x, y);
    //                    }
    //                }
    //            }
    //        }
    //    }
    int startX = 0;
    int startY = 0;
    int endX = targetImage.cols - templateGradX.cols;
    int endY = targetImage.rows - templateGradX.rows;

#pragma omp parallel for collapse(2)
    for (int y = startY; y <= endY; ++y) {
        for (int x = startX; x <= endX; ++x) {
            cv::Mat gradXROI = targetGradX(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));
            cv::Mat gradYROI = targetGradY(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));

            double similarity = computeGradientNormalizedSimilarity1(
                templateData.gradX, templateData.gradY,
                gradXROI, gradYROI,
                edgePoints
            );

#pragma omp critical
            {
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestPos = cv::Point(x, y);
                }
            }
        }
    }
    // 亚像素级 refinement
    const int REFINE_RANGE = 2;
    for (int dy = -REFINE_RANGE; dy <= REFINE_RANGE; ++dy) {
        for (int dx = -REFINE_RANGE; dx <= REFINE_RANGE; ++dx) {
            int x = bestPos.x + dx;
            int y = bestPos.y + dy;
            if (x < 0 || y < 0 ||
                x > targetImage.cols - templateGradX.cols ||
                y > targetImage.rows - templateGradX.rows) continue;

            cv::Mat gradXROI = targetGradX(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));
            cv::Mat gradYROI = targetGradY(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));

            double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
                gradXROI, gradYROI, edgePoints);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestPos = cv::Point(x, y);
            }
        }
    }
    // 三级精细搜索（1/10 → 1/100 → 1/1000）
    cv::Point2f best_position1(bestPos.x, bestPos.y);
    cv::Point2f refined_position = refinePosition1(targetGradX, targetGradY, templateGradX, templateGradY, edgePoints, bestPos, 1.0, 0.1);
    refined_position = refinePosition(targetGradX, targetGradY, templateGradX, templateGradY, edgePoints, refined_position, 0.1, 0.01);
    refined_position = refinePosition(targetGradX, targetGradY, templateGradX, templateGradY, edgePoints, refined_position, 0.01, 0.001);
    return bestPos;
}
// 逐级精细化搜索函数
cv::Point2f FacetEdgeDetector::refinePosition(const cv::Mat& targetGradX, const cv::Mat& targetGradY,
    const cv::Mat& templateGradX, const cv::Mat& templateGradY, const std::vector<cv::Point>& edgePoints, cv::Point2f startPos, double range, double step) {
    cv::Point2f bestPos = startPos;
    double bestScore = -1;

    for (double sub_y = startPos.y - range; sub_y <= startPos.y + range; sub_y += step) {
        for (double sub_x = startPos.x - range; sub_x <= startPos.x + range; sub_x += step) {
            cv::Point2f topLeft(sub_x + templateGradX.cols / 2.0, sub_y + templateGradX.rows / 2.0);

            //cv::Mat targetROI_GradX, targetROI_GradY;
            //cv::getRectSubPix(targetGradX, templateGradX.size(), topLeft, targetROI_GradX);
            //cv::getRectSubPix(targetGradY, templateGradX.size(), topLeft, targetROI_GradY);
            double match_score = computeGradientNormalizedSimilarity0(templateGradX, templateGradY,
                targetGradX, targetGradY, edgePoints, sub_x, sub_y);

            if (match_score > bestScore) {
                bestScore = match_score;
                bestPos = cv::Point2f(sub_x, sub_y);
            }
        }
    }
    return bestPos;

}
// 逐级精细化搜索函数
cv::Point2f FacetEdgeDetector::refinePosition1(const cv::Mat& targetGradX, const cv::Mat& targetGradY,
    const cv::Mat& templateGradX, const cv::Mat& templateGradY, const std::vector<cv::Point>& edgePoints, cv::Point2f startPos, double range, double step) {
    cv::Point2f bestPos = startPos;
    double bestScore = -1;
    auto coeffs_first_call = allCoefficients[0];  // 第一次调用保存的10个Mat
    auto coeffs_second_call = allCoefficients[1]; // 第二次调用保存的10个Mat
    double sumSimilarity = 0.0;
    int validPixels = 0;

    // 第一步：读取sub_edge_points_cropped.txt里的点
    std::vector<cv::Point2f> subedgePoints;
    std::ifstream inFile("sub_edge_points_cropped.txt");
    float readX, readY;
    while (inFile >> readX >> readY) {
        subedgePoints.emplace_back(readX, readY);
    }
    inFile.close();

    for (double sub_y = startPos.y - range; sub_y <= startPos.y + range; sub_y += step) {
        for (double sub_x = startPos.x - range; sub_x <= startPos.x + range; sub_x += step) {

            double sumSimilarity = 0.0;
            int validPixels = 0;

            for (const auto& pt : subedgePoints) {
                float baseX = pt.x;
                float baseY = pt.y;

                // 原始图像上的坐标
                float k2 = coeffs_first_call[1].at<float>(baseY, baseX) - 2 * coeffs_first_call[8].at<float>(baseY, baseX) - coeffs_first_call[6].at<float>(baseY, baseX) * 17 / 5;
                float k3 = coeffs_first_call[2].at<float>(baseY, baseX) - 2 * coeffs_first_call[7].at<float>(baseY, baseX) - coeffs_first_call[9].at<float>(baseY, baseX) * 17 / 5;
                float k4 = coeffs_first_call[3].at<float>(baseY, baseX);
                float k5 = coeffs_first_call[4].at<float>(baseY, baseX);
                float k6 = coeffs_first_call[5].at<float>(baseY, baseX);
                float k7 = coeffs_first_call[6].at<float>(baseY, baseX);
                float k8 = coeffs_first_call[7].at<float>(baseY, baseX);
                float k9 = coeffs_first_call[8].at<float>(baseY, baseX);
                float k10 = coeffs_first_call[9].at<float>(baseY, baseX);

                float gx1 = k2 + 2 * k4 * baseX + k5 * baseY + 3 * k7 * baseX * baseX + 2 * k8 * baseX * baseY + k9 * baseY * baseY;
                float gy1 = k3 + k5 * baseX + 2 * k6 * baseY + k8 * baseX * baseX + 2 * k9 * baseX * baseY + 3 * k10 * baseY * baseY;

                // 偏移之后的坐标
                float newX = baseX + sub_x;
                float newY = baseY + sub_y;

                if (newX < 1 || newY < 1 || newX >= coeffs_second_call[0].cols - 1 || newY >= coeffs_second_call[0].rows - 1)
                    continue;  // 边界检查

                k2 = coeffs_second_call[1].at<float>(newY, newX) - 2 * coeffs_second_call[8].at<float>(newY, newX) - coeffs_second_call[6].at<float>(newY, newX) * 17 / 5;
                k3 = coeffs_second_call[2].at<float>(newY, newX) - 2 * coeffs_second_call[7].at<float>(newY, newX) - coeffs_second_call[9].at<float>(newY, newX) * 17 / 5;
                k4 = coeffs_second_call[3].at<float>(newY, newX);
                k5 = coeffs_second_call[4].at<float>(newY, newX);
                k6 = coeffs_second_call[5].at<float>(newY, newX);
                k7 = coeffs_second_call[6].at<float>(newY, newX);
                k8 = coeffs_second_call[7].at<float>(newY, newX);
                k9 = coeffs_second_call[8].at<float>(newY, newX);
                k10 = coeffs_second_call[9].at<float>(newY, newX);

                float gx2 = k2 + 2 * k4 * newX + k5 * newY + 3 * k7 * newX * newX + 2 * k8 * newX * newY + k9 * newY * newY;
                float gy2 = k3 + k5 * newX + 2 * k6 * newY + k8 * newX * newX + 2 * k9 * newX * newY + 3 * k10 * newY * newY;

                float mag1 = std::sqrt(gx1 * gx1 + gy1 * gy1);
                float mag2 = std::sqrt(gx2 * gx2 + gy2 * gy2);

                if (mag1 > 1e-3 && mag2 > 1e-3) {
                    float sim = (gx1 / mag1) * (gx2 / mag2) + (gy1 / mag1) * (gy2 / mag2);
                    sumSimilarity += sim;
                    validPixels++;
                }
            }

            double match_score = validPixels > 0 ? sumSimilarity / validPixels : 0.0;
            if (match_score > bestScore) {
                bestScore = match_score;
                bestPos = cv::Point2f(sub_x, sub_y);
            }
        }
    }
    return bestPos;

}
cv::Point2f FacetEdgeDetector::refinePosition2(const cv::Mat& targetGradX, const cv::Mat& targetGradY,
    const cv::Mat& templateGradX, const cv::Mat& templateGradY, const std::vector<cv::Point>& edgePoints, cv::Point2f startPos, double range, double step) {
    cv::Point2f bestPos = startPos;
    double bestScore = -1;
    //auto coeffs_first_call = allCoefficients[0];  // 第一次调用保存的10个Mat
    auto coeffs_second_call = allCoefficients[0]; // 第二次调用保存的10个Mat
    double sumSimilarity = 0.0;
    int validPixels = 0;


    std::vector<cv::Point2f> gradientVectors;

    std::ifstream inFile("gradient_vectors1.txt");
    float gx, gy;
    while (inFile >> gx >> gy) {
        gradientVectors.emplace_back(gx, gy);
    }
    inFile.close();
    for (double sub_y = startPos.y - range; sub_y <= startPos.y + range; sub_y += step) {
        for (double sub_x = startPos.x - range; sub_x <= startPos.x + range; sub_x += step) {

            double sumSimilarity = 0.0;
            int validPixels = 0;

            for (size_t i = 0; i < edgePoints.size(); ++i) {
                float baseX = static_cast<float>(edgePoints[i].x);
                float baseY = static_cast<float>(edgePoints[i].y);
                // 从 gradient_vectors.txt 中读取
                double gx1 = gradientVectors[i].x;
                double gy1 = gradientVectors[i].y;

                // 偏移之后的坐标
                float newX = baseX + sub_x;
                float newY = baseY + sub_y;

                if (newX < 1 || newY < 1 || newX >= coeffs_second_call[0].cols - 1 || newY >= coeffs_second_call[0].rows - 1)
                    continue;  // 边界检查

                double k2 = coeffs_second_call[1].at<float>(newY, newX) - 2 * coeffs_second_call[8].at<float>(newY, newX) - coeffs_second_call[6].at<float>(newY, newX) * 17 / 5;
                double k3 = coeffs_second_call[2].at<float>(newY, newX) - 2 * coeffs_second_call[7].at<float>(newY, newX) - coeffs_second_call[9].at<float>(newY, newX) * 17 / 5;
                double k4 = coeffs_second_call[3].at<float>(newY, newX);
                double k5 = coeffs_second_call[4].at<float>(newY, newX);
                double k6 = coeffs_second_call[5].at<float>(newY, newX);
                double k7 = coeffs_second_call[6].at<float>(newY, newX);
                double k8 = coeffs_second_call[7].at<float>(newY, newX);
                double k9 = coeffs_second_call[8].at<float>(newY, newX);
                double k10 = coeffs_second_call[9].at<float>(newY, newX);
                newX = baseX + sub_x;
                newY = baseY + sub_y;

                double gx2 = k2 + 2 * k4 * newX + k5 * newY + 3 * k7 * newX * newX + 2 * k8 * newX * newY + k9 * newY * newY;
                double gy2 = k3 + k5 * newX + 2 * k6 * newY + k8 * newX * newX + 2 * k9 * newX * newY + 3 * k10 * newY * newY;

                double mag1 = std::sqrt(gx1 * gx1 + gy1 * gy1);
                double mag2 = std::sqrt(gx2 * gx2 + gy2 * gy2);

                //if (mag1 > 1e-3 && mag2 > 1e-3) {
                double sim = (gx1 / mag1) * (gx2 / mag2) + (gy1 / mag1) * (gy2 / mag2);
                //double sim = gx1* gx2+ gy1* gy2;
                sumSimilarity += sim;
                validPixels++;
                //}
            }

            double match_score = validPixels > 0 ? sumSimilarity / validPixels : 0.0;
            if (match_score > bestScore) {
                bestScore = match_score;
                bestPos = cv::Point2f(sub_x, sub_y);
            }
        }
    }
    return bestPos;

}

cv::Point2f FacetEdgeDetector::gradientTemplateMatching2(const cv::Mat& templateImage, const cv::Mat& targetImage, cv::Point& best_position, std::vector<cv::Point>& edgePoints, const cv::Mat& templateGradX, const cv::Mat& templateGradY) {
    // 计算模板图像的系数和梯度
    /*cv::Mat templateGradX, templateGradY;
    calculateGradientVectors(templateImage, templateGradX, templateGradY);*/
    //saveGradientVectors(templateImage, templateGradX, templateGradY);
    // 计算目标图像的系数和梯度
    cv::Mat targetGradX, targetGradY;
    calculateGradientVectors(targetImage, targetGradX, targetGradY);
    // 预处理模板数据
    PrecomputedData1 templateData = precomputeTemplateData1(templateGradX, templateGradY);
    float maxX = 0;
    float maxY = 0;

    for (const auto& pt : edgePoints) {
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y > maxY) maxY = pt.y;
    }

    // 图像尺寸应为最大坐标向上取整 + 1，避免越界
    int width = static_cast<int>(std::ceil(maxX)) + 1;
    int height = static_cast<int>(std::ceil(maxY)) + 1;



    // 多尺度搜索参数
    const int SCALES[] = { 4, 2, 1 };  // 多级缩放
    const int STEPS[] = { 8, 4, 1 };  // 各级步长

    cv::Point bestPos(0, 0);
    double maxSimilarity = -1;

    //    for (int scaleIdx = 0; scaleIdx < 3; ++scaleIdx) {
    //        int scale = SCALES[scaleIdx];
    //        int step = STEPS[scaleIdx];
    //
    //        int startX = std::max(0, bestPos.x - 2 * step);
    //        int startY = std::max(0, bestPos.y - 2 * step);
    //        int endX = std::min(targetImage.cols - templateGradX.cols, bestPos.x + 2 * step);
    //        int endY = std::min(targetImage.rows - templateGradX.rows, bestPos.y + 2 * step);
    //
    //        if (scaleIdx == 0) {
    //            startX = 0;
    //            startY = 0;
    //            endX = targetImage.cols - templateGradX.cols;
    //            endY = targetImage.rows - templateGradX.rows;
    //        }
    //
    //#pragma omp parallel for collapse(2)
    //        for (int y = startY; y <= endY; y += step) {
    //            for (int x = startX; x <= endX; x += step) {
    //                cv::Mat gradXROI = targetGradX(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));
    //                cv::Mat gradYROI = targetGradY(cv::Rect(x, y, templateGradY.cols, templateGradY.rows));
    //
    //                double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
    //                    gradXROI, gradYROI, edgePoints);
    //
    //#pragma omp critical
    //                {
    //                    if (similarity > maxSimilarity) {
    //                        maxSimilarity = similarity;
    //                        bestPos = cv::Point(x, y);
    //                    }
    //                }
    //            }
    //        }
    //    }
    std::vector<cv::Point2f> gradientVectors;

    std::ifstream inFile("gradient_vectors.txt");
    float gx, gy;
    while (inFile >> gx >> gy) {
        gradientVectors.emplace_back(gx, gy);
    }
    inFile.close();
    int startX = 0;
    int startY = 0;
    int endX = targetImage.cols - width;
    int endY = targetImage.rows - height;

    int validPixels = edgePoints.size();
    double similarity;
#pragma omp parallel for collapse(2)
    for (int y = startY; y <= endY; ++y) {
        for (int x = startX; x <= endX; ++x) {
            cv::Mat gradXROI = targetGradX(cv::Rect(x, y, width, height));
            cv::Mat gradYROI = targetGradY(cv::Rect(x, y, width, height));
            double sumSimilarity = 0.0;

            for (size_t i = 0; i < edgePoints.size(); ++i) {
                int x = static_cast<int>(edgePoints[i].x);
                int y = static_cast<int>(edgePoints[i].y);

                //if (x >= 0 && x < targetGradX.cols && y >= 0 && y < targetGradY.rows) {// 确保点在图像范围内
                    // 从 gradient_vectors.txt 中读取
                float gx1 = gradientVectors[i].x;
                float gy1 = gradientVectors[i].y;
                float gx2 = gradXROI.at<float>(y, x);
                float gy2 = gradYROI.at<float>(y, x);

                float mag1 = std::sqrt(gx1 * gx1 + gy1 * gy1);
                float mag2 = std::sqrt(gx2 * gx2 + gy2 * gy2);

                //if (mag1 !=0 && mag2 !=0) {  // 只考虑非零梯度的像素
                float gnx1 = gx1 / mag1;
                float gny1 = gy1 / mag1;
                float gnx2 = gx2 / mag2;
                float gny2 = gy2 / mag2;
                float sim = gnx1 * gnx2 + gny1 * gny2;  // 计算余弦相似度

                sumSimilarity += sim;
                validPixels++;
                //}

            //}
            }
            similarity = validPixels > 0 ? sumSimilarity / validPixels : 0.0;

#pragma omp critical
            {
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestPos = cv::Point(x, y);
                }
            }
        }
    }
    //// 亚像素级 refinement
    //const int REFINE_RANGE = 2;
    //for (int dy = -REFINE_RANGE; dy <= REFINE_RANGE; ++dy) {
    //    for (int dx = -REFINE_RANGE; dx <= REFINE_RANGE; ++dx) {
    //        int x = bestPos.x + dx;
    //        int y = bestPos.y + dy;
    //        if (x < 0 || y < 0 ||
    //            x > targetImage.cols - templateGradX.cols ||
    //            y > targetImage.rows - templateGradX.rows) continue;

    //        cv::Mat gradXROI = targetGradX(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));
    //        cv::Mat gradYROI = targetGradY(cv::Rect(x, y, templateGradX.cols, templateGradX.rows));

    //        double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
    //            gradXROI, gradYROI, edgePoints);

    //        if (similarity > maxSimilarity) {
    //            maxSimilarity = similarity;
    //            bestPos = cv::Point(x, y);
    //        }
    //    }
    //}
    // 三级精细搜索（1/10 → 1/100 → 1/1000）
    cv::Point2f best_position1(bestPos.x, bestPos.y);
    cv::Point2f refined_position = refinePosition2(targetGradX, targetGradY, templateGradX, templateGradY, edgePoints, bestPos, 1, 0.1);
    // = refinePosition2(targetGradX, targetGradY, templateGradX, templateGradY, edgePoints, refined_position, 0.1, 0.01);
    // = refinePosition2(targetGradX, targetGradY, templateGradX, templateGradY, edgePoints, refined_position, 0.01, 0.001);
    return bestPos;
}
// 使用梯度向量进行模板匹配
cv::Mat FacetEdgeDetector::gradientTemplateMatching(const cv::Mat& templateImage, const cv::Mat& targetImage, cv::Point& best_position) {
    // 计算模板图像的系数和梯度
    cv::Mat templateGradX, templateGradY;
    calculateGradientVectors(templateImage, templateGradX, templateGradY);

    // 计算目标图像的系数和梯度
    cv::Mat targetGradX, targetGradY;
    calculateGradientVectors(targetImage, targetGradX, targetGradY);

    // 创建结果矩阵（相似度分数）
    //int resultRows = targetImage.rows - templateImage.rows + 1;
    //int resultCols = targetImage.cols - templateImage.cols + 1;
    //cv::Mat resultMatrix = cv::Mat::zeros(resultRows, resultCols, CV_32F);
    /*int resultRows = targetImage.rows - templateImage.rows + 1;
    int resultCols = targetImage.cols - templateImage.cols + 1;*/
    cv::Mat resultMatrix = cv::Mat::zeros(20, 20, CV_32F);
    // 归一化模板梯度
    cv::Mat templateGradMagnitude;
    cv::magnitude(templateGradX, templateGradY, templateGradMagnitude);
    cv::Mat templateNormX, templateNormY;

    // 避免除以零
    cv::Mat nonZeroMask = templateGradMagnitude > 1e-6;
    templateNormX = cv::Mat::zeros(templateGradX.size(), CV_32F);
    templateNormY = cv::Mat::zeros(templateGradY.size(), CV_32F);

    templateGradX.copyTo(templateNormX, nonZeroMask);
    templateGradY.copyTo(templateNormY, nonZeroMask);

    cv::divide(templateNormX, templateGradMagnitude, templateNormX, 1.0, -1);
    cv::divide(templateNormY, templateGradMagnitude, templateNormY, 1.0, -1);
    std::vector<std::pair<double, int>> match_val;
    std::vector<std::pair<int, cv::Point2f>> match_loc;
    int  best_index;
    cv::Point2f best_loc;
    // 对每个可能的位置进行匹配计算
    //for (int y = best_position.y-1; y < best_position.y+1; y=y+0.1) {
    //    for (int x = best_position.x - 1; x < best_position.x+1; x=x+0.1) {
    //        // 提取当前位置的目标图像区域
    //        //cv::Rect roi(x, y, templateImage.cols, templateImage.rows);
    //        cv::Rect roi(best_position.x - templateImage.cols / 2, best_position.y - templateImage.rows / 2, templateImage.cols, templateImage.rows);
    //        cv::Mat targetROI_GradX = targetGradX(roi);
    //        cv::Mat targetROI_GradY = targetGradY(roi);

    //        // 计算目标区域的梯度幅值
    //        cv::Mat targetGradMagnitude;
    //        cv::magnitude(targetROI_GradX, targetROI_GradY, targetGradMagnitude);

    //        // 归一化目标梯度
    //        cv::Mat targetNormX = cv::Mat::zeros(targetROI_GradX.size(), CV_32F);
    //        cv::Mat targetNormY = cv::Mat::zeros(targetROI_GradY.size(), CV_32F);

    //        cv::Mat targetNonZeroMask = targetGradMagnitude > 1e-6;
    //        targetROI_GradX.copyTo(targetNormX, targetNonZeroMask);
    //        targetROI_GradY.copyTo(targetNormY, targetNonZeroMask);

    //        cv::divide(targetNormX, targetGradMagnitude, targetNormX, 1.0, -1);
    //        cv::divide(targetNormY, targetGradMagnitude, targetNormY, 1.0, -1);

    //        // 计算梯度方向的点积 (cos(theta))
    //        cv::Mat dotProduct = targetNormX.mul(templateNormX) + targetNormY.mul(templateNormY);

    //        // 计算有效像素数
    //        cv::Mat validMask = nonZeroMask & targetNonZeroMask;
    //        int validPixels = cv::countNonZero(validMask);

    //        // 计算相似度分数（归一化点积和）
    //        if (validPixels > 0) {
    //            float similarity = cv::sum(dotProduct)[0] / validPixels;
    //            //resultMatrix.at<float>(y, x) = similarity;
    //            match_val.emplace_back(similarity, match_val.size());
    //            match_loc.emplace_back(match_loc.size(), cv::Point(static_cast<double>(x), static_cast<double>(y)));
    //        }
    //    }
    //}
#pragma omp parallel for collapse(2)
    for (int y = best_position.y - 1; y <= best_position.y + 1; y++) {
        for (int x = best_position.x - 1; x <= best_position.x + 1; x++) {
            for (double sub_y = y; sub_y < y + 1; sub_y += 0.1) {
                for (double sub_x = x; sub_x < x + 1; sub_x += 0.1) {
                    // 通过 cv::getRectSubPix 获取亚像素梯度区域
                    cv::Mat targetROI_GradX, targetROI_GradY;
                    cv::getRectSubPix(targetGradX, templateImage.size(), cv::Point2f(sub_x, sub_y), targetROI_GradX);
                    cv::getRectSubPix(targetGradY, templateImage.size(), cv::Point2f(sub_x, sub_y), targetROI_GradY);

                    // 计算梯度幅值
                    cv::Mat targetGradMagnitude;
                    cv::magnitude(targetROI_GradX, targetROI_GradY, targetGradMagnitude);

                    // 梯度方向归一化
                    cv::Mat targetNormX, targetNormY;
                    cv::divide(targetROI_GradX, targetGradMagnitude, targetNormX, 1.0, CV_32F);
                    cv::divide(targetROI_GradY, targetGradMagnitude, targetNormY, 1.0, CV_32F);

                    // 计算点积（余弦相似度）
                    cv::Mat dotProduct = targetNormX.mul(templateNormX) + targetNormY.mul(templateNormY);

                    // 有效像素判断
                    cv::Mat validMask = nonZeroMask & (targetGradMagnitude > 1e-6);
                    int validPixels = cv::countNonZero(validMask);

                    // 计算相似度
                    if (validPixels > 0) {
                        float similarity = cv::sum(dotProduct)[0] / validPixels;

#pragma omp critical
                        {
                            match_val.emplace_back(similarity, match_val.size());
                            match_loc.emplace_back(match_loc.size(), cv::Point2f(static_cast<double>(sub_x), static_cast<double>(sub_y)));
                        }
                    }
                }
            }
        }
    }
    auto max_iter = std::max_element(match_val.begin(), match_val.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

    best_index = max_iter->second;
    best_loc = match_loc[best_index].second;




    return resultMatrix;
}


