
#include <opencv2/opencv.hpp>
#include <iostream>
#include "sim_template.h"
#include <omp.h>
#include <fstream>

using namespace cv;
using namespace std;
// 计算梯度并归一化
void computeGradient(const Mat& img, Mat& gradX, Mat& gradY, Mat& magnitude, Mat& direction) {
    // 确保输入是灰度图
    CV_Assert(img.channels() == 1);

    // 计算 Sobel 梯度
    Sobel(img, gradX, CV_32F, 1, 0, 3);
    Sobel(img, gradY, CV_32F, 0, 1, 3);

    // 初始化结果矩阵
    magnitude = Mat(img.size(), CV_32F);
    direction = Mat(img.size(), CV_32F);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            float gx = gradX.at<float>(y, x);
            float gy = gradY.at<float>(y, x);

            // 计算梯度幅值
            float mag = std::sqrt(gx * gx + gy * gy);
            magnitude.at<float>(y, x) = mag;

            // 计算梯度方向（弧度制，范围 -π ~ π）
            direction.at<float>(y, x) = std::atan2(gy, gx);
        }
    }
}
// 计算模板匹配相似性 S
double computeSimilarity(const cv::Mat& templateDir, const cv::Mat& imageDir, cv::Point matchPoint) {
    int rows = templateDir.rows;
    int cols = templateDir.cols;
    double similarity = 0.0;

    // 边界检查，防止越界访问
    if (matchPoint.y + rows > imageDir.rows || matchPoint.x + cols > imageDir.cols) {
        return -1.0;  // 返回无效匹配值
    }

    // 确保数据类型正确
    if (templateDir.type() != CV_32FC2 || imageDir.type() != CV_32FC2) {
        cerr << "Error: templateDir or imageDir is not CV_32FC2!" << endl;
        return -1.0;
    }

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            Vec2f tGrad = templateDir.at<Vec2f>(y, x);
            Vec2f iGrad = imageDir.at<Vec2f>(matchPoint.y + y, matchPoint.x + x);

            // 过滤掉 (0,0) 向量
            if (tGrad[0] == 0.0f && tGrad[1] == 0.0f) continue;
            if (iGrad[0] == 0.0f && iGrad[1] == 0.0f) continue;

            double dotProduct = tGrad[0] * iGrad[0] + tGrad[1] * iGrad[1];  // 归一化点积
            similarity += dotProduct;
        }
    }

    return similarity / (rows * cols);
}


// 在待匹配图像中搜索最佳匹配位置
cv::Point templateMatching(const cv::Mat& image, const cv::Mat& templ) {
    cv::Mat imageGradX, imageGradY, imageMag, imageDir;
    cv::Mat templGradX, templGradY, templMag, templDir;

    // 计算梯度
    computeGradient(image, imageGradX, imageGradY, imageMag, imageDir);
    computeGradient(templ, templGradX, templGradY, templMag, templDir);

    cv::Point bestMatch(0, 0);
    double bestScore = -1.0;

    // 遍历搜索
#pragma omp parallel for collapse(2)
    for (int y = 0; y <= image.rows - templ.rows; y++) {
        for (int x = 0; x <= image.cols - templ.cols; x++) {
            double score = computeSimilarity(templDir, imageDir, Point(x, y));
            if (score > bestScore) {
                bestScore = score;
                bestMatch = Point(x, y);
            }
        }
    }

    return bestMatch;
}

// 计算灰度图的 Sobel 梯度
void computeGradient(const cv::Mat& img, cv::Mat& gradX, cv::Mat& gradY) {
    Sobel(img, gradX, CV_32F, 1, 0, 3);
    Sobel(img, gradY, CV_32F, 0, 1, 3);
}

// 计算两个梯度矩阵的归一化相似度
double computeGradientNormalizedSimilarity(const cv::Mat& gradX1, const cv::Mat& gradY1,
    const cv::Mat& gradX2, const cv::Mat& gradY2) {
    CV_Assert(gradX1.size() == gradX2.size() && gradY1.size() == gradY2.size());

    double sumSimilarity = 0.0;
    int validPixels = 0;

    for (int y = 0; y < gradX1.rows; y++) {
        for (int x = 0; x < gradX1.cols; x++) {
            float gx1 = gradX1.at<float>(y, x);
            float gy1 = gradY1.at<float>(y, x);
            float gx2 = gradX2.at<float>(y, x);
            float gy2 = gradY2.at<float>(y, x);

            float mag1 = sqrt(gx1 * gx1 + gy1 * gy1);
            float mag2 = sqrt(gx2 * gx2 + gy2 * gy2);

            float gnx1 = gx1 / (mag1 + 1e-6);
            float gny1 = gy1 / (mag1 + 1e-6);
            float gnx2 = gx2 / (mag2 + 1e-6);
            float gny2 = gy2 / (mag2 + 1e-6);

            float sim = gnx1 * gnx2 + gny1 * gny2;

            if (mag1 > 1e-3 && mag2 > 1e-3) {
                sumSimilarity += sim;
                validPixels++;
            }
        }
    }

    return validPixels > 0 ? sumSimilarity / validPixels : 0.0;
}

// **基于梯度归一化相似度的模板匹配**
cv::Point matchTemplateGradient(const cv::Mat& image, const cv::Mat& templateImg) {
    cv::Mat gradX1, gradY1, gradX2, gradY2;
    computeGradient(templateImg, gradX1, gradY1);
    computeGradient(image, gradX2, gradY2);

    int bestX = 0, bestY = 0;
    double maxSimilarity = -1.0;

    int stepX = 3;  // 可调整步长，提高匹配速度
    int stepY = 3;

    // 遍历搜索窗口
    for (int y = 0; y <= image.rows - templateImg.rows; y += stepY) {
        for (int x = 0; x <= image.cols - templateImg.cols; x += stepX) {
            cv::Rect roi(x, y, templateImg.cols, templateImg.rows);
            cv::Mat searchGradX = gradX2(roi);
            cv::Mat searchGradY = gradY2(roi);

            double similarity = computeGradientNormalizedSimilarity(gradX1, gradY1, searchGradX, searchGradY);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestX = x;
                bestY = y;
            }
        }
    }

    cout << "最佳匹配位置: (" << bestX << ", " << bestY << "), 相似度: " << maxSimilarity << endl;
    return cv::Point(bestX, bestY);
}
cv::Point matchTemplateGradient(const cv::Mat& image, const cv::Mat& gradX1, const cv::Mat& gradY1) {
    cv::Mat gradX2, gradY2;

    computeGradient(image, gradX2, gradY2);

    int bestX = 0, bestY = 0;
    double maxSimilarity = -1.0;

    int stepX = 3;  // 可调整步长，提高匹配速度
    int stepY = 3;

    // 遍历搜索窗口
    for (int y = 0; y <= image.rows - gradX1.rows; y += stepY) {
        for (int x = 0; x <= image.cols - gradX1.cols; x += stepX) {
            cv::Rect roi(x, y, gradX1.cols, gradX1.rows);
            cv::Mat searchGradX = gradX2(roi);
            cv::Mat searchGradY = gradY2(roi);

            double similarity = computeGradientNormalizedSimilarity(gradX1, gradY1, searchGradX, searchGradY);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestX = x;
                bestY = y;
            }
        }
    }

    cout << "最佳匹配位置: (" << bestX << ", " << bestY << "), 相似度: " << maxSimilarity << endl;
    return cv::Point(bestX, bestY);
}
// 计算边缘点和它们的梯度
void extractEdgePointsWithGradient(const cv::Mat& img, std::vector<cv::Point>& edgePoints, std::vector<float>& gradientX, std::vector<float>& gradientY) {


    // 计算Sobel梯度
    cv::Mat gradX, gradY;
    Sobel(img, gradX, CV_32F, 1, 0, 3);
    Sobel(img, gradY, CV_32F, 0, 1, 3);



    // 遍历边缘点，提取梯度信息
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.at<uchar>(y, x) > 0) {  // 只考虑Canny检测出的边缘点
                float gx = gradX.at<float>(y, x);
                float gy = gradY.at<float>(y, x);

                edgePoints.push_back(cv::Point(x, y));
                gradientX.push_back(gx);
                gradientY.push_back(gy);
            }
        }
    }
}

// 将边缘点的梯度值存入原图大小的图像
void saveGradientToImage(const std::vector<cv::Point>& edgePoints, const std::vector<float>& gradientX, const std::vector<float>& gradientY, cv::Mat& gradImageX, cv::Mat& gradImageY) {
    gradImageX = cv::Mat::zeros(gradImageX.size(), CV_32F);
    gradImageY = cv::Mat::zeros(gradImageY.size(), CV_32F);

    for (size_t i = 0; i < edgePoints.size(); i++) {
        cv::Point pt = edgePoints[i];
        gradImageX.at<float>(pt.y, pt.x) = gradientX[i];
        gradImageY.at<float>(pt.y, pt.x) = gradientY[i];
    }
}

// 裁剪边缘点区域
void cropEdgeRegion(const cv::Mat& img, const cv::Mat& edgeImg, const std::vector<cv::Point>& edgePoints, cv::Mat& gradImageX, cv::Mat& gradImageY, cv::Mat& croppedGradX, cv::Mat& croppedGradY, cv::Mat& croppededge, cv::Mat& croppedImg) {
    // 寻找边缘点区域的最小边界框
    int minX = edgePoints[0].x, minY = edgePoints[0].y;
    int maxX = edgePoints[0].x, maxY = edgePoints[0].y;

    for (size_t i = 1; i < edgePoints.size(); i++) {
        minX = min(minX, edgePoints[i].x);
        minY = min(minY, edgePoints[i].y);
        maxX = max(maxX, edgePoints[i].x);
        maxY = max(maxY, edgePoints[i].y);
    }

    // 裁剪出边缘点区域
    Rect roi(minX, minY, maxX - minX + 1, maxY - minY + 1);
    croppedGradX = gradImageX(roi);
    croppedGradY = gradImageY(roi);
    croppededge = edgeImg(roi);
    croppedImg = img(roi);
    // 图像尺寸（可选，防止越界）
    int croppedWidth = roi.width;
    int croppedHeight = roi.height;

    // 读取 sub_edge_points.txt
    std::ifstream inFile("sub_edge_points.txt");
    std::ofstream outFile("sub_edge_points_cropped.txt");

    float x, y;
    while (inFile >> x >> y) {
        float croppedX = x - roi.x;
        float croppedY = y - roi.y;

        // 过滤掉不在裁剪区域内的点
        if (croppedX >= 0 && croppedX < croppedWidth &&
            croppedY >= 0 && croppedY < croppedHeight) {
            outFile << croppedX << " " << croppedY << std::endl;
        }
    }

    inFile.close();
    outFile.close();
}


// 预计算模板梯度信息
struct PrecomputedData {
    Mat gradX, gradY; // 存储梯度信息
    Size templateSize;
};

PrecomputedData precomputeTemplateData(const Mat& gradX1, const Mat& gradY1) {
    PrecomputedData data;
    data.gradX = gradX1.clone();
    data.gradY = gradY1.clone();
    data.templateSize = gradX1.size();
    return data;
}

// 计算两个梯度矩阵的归一化相似度
double computeGradientNormalizedSimilarity1(const  cv::Mat& gradX1, const  cv::Mat& gradY1,
    const  cv::Mat& gradX2, const  cv::Mat& gradY2, const std::vector<cv::Point>& edgePoints) {
    CV_Assert(gradX1.size() == gradX2.size() && gradY1.size() == gradY2.size());

    double sumSimilarity = 0.0;
    int validPixels = 0;

    for (const auto& pt : edgePoints) {
        int x = pt.x;
        int y = pt.y;

        if (x >= 0 && x < gradX1.cols && y >= 0 && y < gradX1.rows) { // 确保点在图像范围内
            float gx1 = gradX1.at<float>(y, x);
            float gy1 = gradY1.at<float>(y, x);
            float gx2 = gradX2.at<float>(y, x);
            float gy2 = gradY2.at<float>(y, x);

            float mag1 = std::sqrt(gx1 * gx1 + gy1 * gy1);
            float mag2 = std::sqrt(gx2 * gx2 + gy2 * gy2);

            //if (mag1 > 1e-3 && mag2 > 1e-3) {  // 只考虑非零梯度的像素
            float gnx1 = gx1 / mag1;
            float gny1 = gy1 / mag1;
            float gnx2 = gx2 / mag2;
            float gny2 = gy2 / mag2;
            float sim = gnx1 * gnx2 + gny1 * gny2;  // 计算余弦相似度

            sumSimilarity += sim;
            validPixels++;
            //}
        }
    }

    return validPixels > 0 ? sumSimilarity / validPixels : 0.0;
}
double computeGradientNormalizedSimilarity0(const  cv::Mat& gradX1, const  cv::Mat& gradY1,
    const  cv::Mat& gradX2, const  cv::Mat& gradY2, const std::vector<cv::Point>& edgePoints, double subx, double suby) {
    //CV_Assert(gradX1.size() == gradX2.size() && gradY1.size() == gradY2.size());

    double sumSimilarity = 0.0;
    int validPixels = 0;

    for (const auto& pt : edgePoints) {
        int x = pt.x;
        int y = pt.y;

        if (x >= 0 && x < gradX1.cols && y >= 0 && y < gradX1.rows) { // 确保点在图像范围内
            float gx1 = gradX1.at<float>(y, x);
            float gy1 = gradY1.at<float>(y, x);
            float gx2 = gradX2.at<float>(y + suby, x + subx);
            float gy2 = gradY2.at<float>(y + suby, x + subx);

            float mag1 = std::sqrt(gx1 * gx1 + gy1 * gy1);
            float mag2 = std::sqrt(gx2 * gx2 + gy2 * gy2);

            if (mag1 > 1e-3 && mag2 > 1e-3) {  // 只考虑非零梯度的像素
                float gnx1 = gx1 / mag1;
                float gny1 = gy1 / mag1;
                float gnx2 = gx2 / mag2;
                float gny2 = gy2 / mag2;
                float sim = gnx1 * gnx2 + gny1 * gny2;  // 计算余弦相似度

                sumSimilarity += sim;
                validPixels++;
            }
        }
    }

    return validPixels > 0 ? sumSimilarity / validPixels : 0.0;
}
// 进行模板匹配
cv::Point matchTemplateGradientOptimized(const cv::Mat& image,
    const cv::Mat& gradX1,
    const cv::Mat& gradY1, const std::vector<cv::Point>& edgePoints) {
    // 预处理模板数据
    PrecomputedData templateData = precomputeTemplateData(gradX1, gradY1);

    // 计算图像梯度
    cv::Mat gradX2, gradY2;
    computeGradient(image, gradX2, gradY2);

    // 多尺度搜索参数
    const int SCALES[] = { 4, 2, 1 };  // 多级缩放
    const int STEPS[] = { 8, 4, 1 };  // 各级步长

    Point bestPos(0, 0);
    double maxSimilarity = -1;
    int startX = 0;
    int startY = 0;
    int endX = image.cols - gradX1.cols;
    int endY = image.rows - gradX1.rows;

#pragma omp parallel for collapse(2)
    for (int y = startY; y <= endY; ++y) {
        for (int x = startX; x <= endX; ++x) {
            cv::Mat gradXROI = gradX2(cv::Rect(x, y, gradX1.cols, gradX1.rows));
            cv::Mat gradYROI = gradY2(cv::Rect(x, y, gradX1.cols, gradX1.rows));

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

    /*for (int scaleIdx = 0; scaleIdx < 3; ++scaleIdx) {
        int scale = SCALES[scaleIdx];
        int step = STEPS[scaleIdx];

        int startX = max(0, bestPos.x - 2 * step);
        int startY = max(0, bestPos.y - 2 * step);
        int endX = min(image.cols - gradX1.cols, bestPos.x + 2 * step);
        int endY = min(image.rows - gradX1.rows, bestPos.y + 2 * step);

        if (scaleIdx == 0) {
            startX = 0;
            startY = 0;
            endX = image.cols - gradX1.cols;
            endY = image.rows - gradX1.rows;
        }

#pragma omp parallel for collapse(2)
        for (int y = startY; y <= endY; y += step) {
            for (int x = startX; x <= endX; x += step) {
                Mat gradXROI = gradX2(Rect(x, y, gradX1.cols, gradX1.rows));
                Mat gradYROI = gradY2(Rect(x, y, gradX1.cols, gradX1.rows));

                double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
                    gradXROI, gradYROI, edgePoints);

#pragma omp critical
                {
                    if (similarity > maxSimilarity) {
                        maxSimilarity = similarity;
                        bestPos = Point(x, y);
                    }
                }
            }
        }
    }*/

    // 亚像素级 refinement
    const int REFINE_RANGE = 2;
    for (int dy = -REFINE_RANGE; dy <= REFINE_RANGE; ++dy) {
        for (int dx = -REFINE_RANGE; dx <= REFINE_RANGE; ++dx) {
            int x = bestPos.x + dx;
            int y = bestPos.y + dy;
            if (x < 0 || y < 0 ||
                x > image.cols - gradX1.cols ||
                y > image.rows - gradX1.rows) continue;

            Mat gradXROI = gradX2(Rect(x, y, gradX1.cols, gradX1.rows));
            Mat gradYROI = gradY2(Rect(x, y, gradX1.cols, gradX1.rows));

            double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
                gradXROI, gradYROI, edgePoints);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestPos = Point(x, y);
            }
        }
    }

    cout << "Optimized最佳匹配: (" << bestPos.x << ", " << bestPos.y
        << "), 相似度: " << maxSimilarity << endl;
    return bestPos;
}
cv::Point matchTemplateGradientOptimized1(const cv::Mat& image,
    const cv::Mat& gradX1,
    const cv::Mat& gradY1, const std::vector<cv::Point>& edgePoints) {

    // 预处理模板数据
    PrecomputedData templateData = precomputeTemplateData(gradX1, gradY1);

    // 计算图像梯度
    cv::Mat gradX2, gradY2;
    computeGradient(image, gradX2, gradY2);

    // **1. 计算边缘点的重心**
    cv::Point2f center(0, 0);
    for (const auto& pt : edgePoints) {
        center.x += pt.x;
        center.y += pt.y;
    }
    center.x /= edgePoints.size();
    center.y /= edgePoints.size();

    cv::Point bestPos(center.x, center.y); // **以重心作为搜索起点**
    bestPos.x = std::max(0, std::min(bestPos.x, image.cols - gradX1.cols));
    bestPos.y = std::max(0, std::min(bestPos.y, image.rows - gradX1.rows));

    double maxSimilarity = -1;

    // **2. 多尺度搜索参数**
    const int SCALES[] = { 4, 2, 1 };  // 多级缩放
    const int STEPS[] = { 8, 4, 1 };   // 各级步长

    for (int scaleIdx = 0; scaleIdx < 3; ++scaleIdx) {
        int scale = SCALES[scaleIdx];
        int step = STEPS[scaleIdx];

        int startX = std::max(0, bestPos.x - 2 * step);
        int startY = std::max(0, bestPos.y - 2 * step);
        int endX = std::min(image.cols - gradX1.cols, bestPos.x + 2 * step);
        int endY = std::min(image.rows - gradX1.rows, bestPos.y + 2 * step);

        if (scaleIdx == 0) {  // **第一轮全局搜索**
            startX = 0;
            startY = 0;
            endX = image.cols - gradX1.cols;
            endY = image.rows - gradX1.rows;
        }

#pragma omp parallel for collapse(2)
        for (int y = startY; y <= endY; y += step) {
            for (int x = startX; x <= endX; x += step) {
                cv::Mat gradXROI = gradX2(cv::Rect(x, y, gradX1.cols, gradX1.rows));
                cv::Mat gradYROI = gradY2(cv::Rect(x, y, gradX1.cols, gradX1.rows));

                double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
                    gradXROI, gradYROI, edgePoints);

#pragma omp critical
                {
                    if (similarity > maxSimilarity) {
                        maxSimilarity = similarity;
                        bestPos = cv::Point(x, y);
                    }
                }
            }
        }
    }

    // **3. 亚像素级 Refinement**
    const int REFINE_RANGE = 2;
    for (int dy = -REFINE_RANGE; dy <= REFINE_RANGE; ++dy) {
        for (int dx = -REFINE_RANGE; dx <= REFINE_RANGE; ++dx) {
            int x = bestPos.x + dx;
            int y = bestPos.y + dy;
            if (x < 0 || y < 0 || x > image.cols - gradX1.cols || y > image.rows - gradX1.rows) continue;

            cv::Mat gradXROI = gradX2(cv::Rect(x, y, gradX1.cols, gradX1.rows));
            cv::Mat gradYROI = gradY2(cv::Rect(x, y, gradX1.cols, gradX1.rows));

            double similarity = computeGradientNormalizedSimilarity1(templateData.gradX, templateData.gradY,
                gradXROI, gradYROI, edgePoints);

            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestPos = cv::Point(x, y);
            }
        }
    }
    cout << "Optimized最佳匹配: (" << bestPos.x << ", " << bestPos.y
        << "), 相似度: " << maxSimilarity << endl;
    return bestPos;
}
void drawTemplateEdges(cv::Mat& srcImage, std::vector<cv::Point>& edgePoints, cv::Point position) {
    cv::Mat colorImage;
    if (srcImage.channels() == 1) {
        cv::cvtColor(srcImage, colorImage, cv::COLOR_GRAY2BGR);
    }
    else {
        colorImage = srcImage.clone(); // 直接使用原图
    }
    std::vector<cv::Point> transformedPoints;
    for (const auto& pt : edgePoints) {
        transformedPoints.emplace_back(pt.x + position.x, pt.y + position.y);
    }

    for (const auto& pt : transformedPoints) {
        cv::circle(colorImage, pt, 0.5, cv::Scalar(0, 0, 255), -1); // 红色点
    }

    //cv::polylines(srcImage, transformedPoints, true, cv::Scalar(0, 255, 0), 1); // 绿色线条
}

