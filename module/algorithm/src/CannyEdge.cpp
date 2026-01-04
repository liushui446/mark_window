// CannyEdgeDetection.cpp
#include "CannyEdge.h"
#include <numeric>
cv::Mat CannyEdge::detectEdges(const cv::Mat& inputImage,std::vector<cv::Point>& edgePoints, int&binarize_thresh) {
    // Convert to grayscale if not already
    cv::Mat grayImage;
    if (inputImage.channels() > 1) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = inputImage.clone();
    }
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

    // 形态学操作：去除小的杂点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // 先开运算去除小的白点噪声
    cv::Mat openedEdges;
    cv::morphologyEx(edges2, openedEdges, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
     //使用连通域分析去除小杂点
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(edges2, labels, stats, centroids);

    // 生成一个新的掩码，只保留较大的连通域
    cv::Mat filteredEdges = cv::Mat::zeros(edges2.size(), CV_8UC1);
    for (int i = 1; i < numLabels; i++) { // 跳过背景标签0
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > 250) { // 设定一个面积阈值（可以调整）
            filteredEdges.setTo(255, labels == i);
        }
    }
    cv::Mat binary_img, edges3;
    cv::threshold(grayImage, binary_img, binarize_thresh, 255, cv::THRESH_BINARY);
    cv::Canny(binary_img, edges3, 140, 200, 3, true);
    // 去除小的边缘碎片
    /*cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(edges3, labels, stats, centroids, 8, CV_32S);
    cv::Mat cleaned_edges = cv::Mat::zeros(edges3.size(), CV_8UC1);

    const int min_area = 30;
    for (int i = 1; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_area) {
            cleaned_edges.setTo(255, labels == i);
        }
    }*/
    // Convert to floating point for processing
    grayImage.convertTo(grayImage, CV_32F);

    //// Step 1: 高斯过滤
    ////cv::Mat smoothedImage = applyGaussianFilter(grayImage);
    //cv::Mat smoothedImage = smoothedImage2.clone();
    //// Step 2: 计算梯度
    //cv::Mat gradientMagnitude, gradientX, gradientY;
    //computeSobelGradients(smoothedImage, gradientMagnitude, gradientX, gradientY);

    //// Step 3: 非极大值抑制
    //cv::Mat nonMaxSuppressed = nonMaximumSuppression(gradientMagnitude, gradientX, gradientY);

    //// Step 4:滞后阈值处理
    //cv::Mat edges = hysteresisThresholding(nonMaxSuppressed, 100, 120);
   
    /*for (int y = 0; y < cleaned_edges.rows; y++) {
        for (int x = 0; x < cleaned_edges.cols; x++) {
            if (cleaned_edges.at<uchar>(y, x) > 0) {
                edgePoints.push_back(cv::Point(x, y));
            }
        }
    }*/

    for (int y = 0; y < edges3.rows; y++) {
        for (int x = 0; x < edges3.cols; x++) {
        if (edges3.at<uchar>(y, x) > 0) {
            edgePoints.push_back(cv::Point(x, y));
        }
    }
}
    //return edges2;
    return edges3;
}
cv::Mat CannyEdge::detectEdges(const cv::Mat& inputImage, std::vector<cv::Point>& edgePoints) {
    // Convert to grayscale if not already
    cv::Mat grayImage;
    if (inputImage.channels() > 1) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = inputImage.clone();
    }
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

    // 形态学操作：去除小的杂点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // 先开运算去除小的白点噪声
    cv::Mat openedEdges;
    cv::morphologyEx(edges2, openedEdges, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    //使用连通域分析去除小杂点
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(edges2, labels, stats, centroids);

    // 生成一个新的掩码，只保留较大的连通域
    cv::Mat filteredEdges = cv::Mat::zeros(edges2.size(), CV_8UC1);
    for (int i = 1; i < numLabels; i++) { // 跳过背景标签0
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > 250) { // 设定一个面积阈值（可以调整）
            filteredEdges.setTo(255, labels == i);
        }
    }
    cv::Mat binary_img, edges3;
    cv::threshold(grayImage, binary_img,160, 255, cv::THRESH_BINARY);
    cv::Canny(binary_img, edges3, 140, 200, 3, true);
    // 去除小的边缘碎片
    /*cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(edges3, labels, stats, centroids, 8, CV_32S);
    cv::Mat cleaned_edges = cv::Mat::zeros(edges3.size(), CV_8UC1);

    const int min_area = 30;
    for (int i = 1; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= min_area) {
            cleaned_edges.setTo(255, labels == i);
        }
    }*/
    // Convert to floating point for processing
    grayImage.convertTo(grayImage, CV_32F);

    //// Step 1: 高斯过滤
    ////cv::Mat smoothedImage = applyGaussianFilter(grayImage);
    //cv::Mat smoothedImage = smoothedImage2.clone();
    //// Step 2: 计算梯度
    //cv::Mat gradientMagnitude, gradientX, gradientY;
    //computeSobelGradients(smoothedImage, gradientMagnitude, gradientX, gradientY);

    //// Step 3: 非极大值抑制
    //cv::Mat nonMaxSuppressed = nonMaximumSuppression(gradientMagnitude, gradientX, gradientY);

    //// Step 4:滞后阈值处理
    //cv::Mat edges = hysteresisThresholding(nonMaxSuppressed, 100, 120);

    /*for (int y = 0; y < cleaned_edges.rows; y++) {
        for (int x = 0; x < cleaned_edges.cols; x++) {
            if (cleaned_edges.at<uchar>(y, x) > 0) {
                edgePoints.push_back(cv::Point(x, y));
            }
        }
    }*/

    for (int y = 0; y < edges3.rows; y++) {
        for (int x = 0; x < edges3.cols; x++) {
            if (edges3.at<uchar>(y, x) > 0) {
                edgePoints.push_back(cv::Point(x, y));
            }
        }
    }
    //return edges2;
    return edges3;
}

cv::Mat CannyEdge::applyGaussianFilter(const cv::Mat& image) {
    cv::Mat smoothedImage;
    cv::GaussianBlur(image, smoothedImage, cv::Size(5, 5), 0);
    return smoothedImage;
}

void CannyEdge::computeSobelGradients(const cv::Mat& image,
    cv::Mat& gradientMagnitude,
    cv::Mat& gradientX,
    cv::Mat& gradientY) {
    // 使用sobel算子计算 x and y gradients 
    cv::Sobel(image, gradientX, CV_32F, 1, 0, 3);
    cv::Sobel(image, gradientY, CV_32F, 0, 1, 3);

    // 计算梯度方向 使用 sqrt(Gx^2 + Gy^2)
    cv::magnitude(gradientX, gradientY, gradientMagnitude);
}

cv::Mat CannyEdge::nonMaximumSuppression(const cv::Mat& gradientMagnitude,
    const cv::Mat& gradientX,
    const cv::Mat& gradientY) {
    cv::Mat suppressed = gradientMagnitude.clone();
    suppressed.setTo(0);

    for (int y = 1; y < gradientMagnitude.rows - 1; y++) {
        for (int x = 1; x < gradientMagnitude.cols - 1; x++) {
            float dx = gradientX.at<float>(y, x);
            float dy = gradientY.at<float>(y, x);
            float magnitude = gradientMagnitude.at<float>(y, x);

            // 计算梯度方向
            float angle = std::atan2(dy, dx);
            angle = angle * 180.0 / CV_PI;
            if (angle < 0) angle += 180;

            //根据梯度方向检查邻域
            bool keepPixel = false;
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                // Horizontal edge
                keepPixel = (magnitude >= gradientMagnitude.at<float>(y, x - 1) &&
                    magnitude >= gradientMagnitude.at<float>(y, x + 1));
            }
            else if (angle >= 22.5 && angle < 67.5) {
                // Diagonal (bottom-left to top-right)
                keepPixel = (magnitude >= gradientMagnitude.at<float>(y - 1, x + 1) &&
                    magnitude >= gradientMagnitude.at<float>(y + 1, x - 1));
            }
            else if (angle >= 67.5 && angle < 112.5) {
                // Vertical edge
                keepPixel = (magnitude >= gradientMagnitude.at<float>(y - 1, x) &&
                    magnitude >= gradientMagnitude.at<float>(y + 1, x));
            }
            else if (angle >= 112.5 && angle < 157.5) {
                // Diagonal (top-left to bottom-right)
                keepPixel = (magnitude >= gradientMagnitude.at<float>(y - 1, x - 1) &&
                    magnitude >= gradientMagnitude.at<float>(y + 1, x + 1));
            }

            if (keepPixel) {
                suppressed.at<float>(y, x) = magnitude;
            }
        }
    }

    return suppressed;
}

cv::Mat CannyEdge::hysteresisThresholding(const cv::Mat& nonMaxSuppressed,
    double lowThreshold,
    double highThreshold) {
    cv::Mat edges = cv::Mat::zeros(nonMaxSuppressed.size(), CV_8U);

    for (int y = 0; y < nonMaxSuppressed.rows; y++) {
        for (int x = 0; x < nonMaxSuppressed.cols; x++) {
            if (nonMaxSuppressed.at<float>(y, x) > highThreshold) {
                edges.at<uchar>(y, x) = 255;
                recursiveHysteresis(edges, nonMaxSuppressed, x, y, lowThreshold);
            }
        }
    }

    return edges;
}

void CannyEdge::recursiveHysteresis(cv::Mat& edges, const cv::Mat& nonMaxSuppressed, int x, int y, double lowThreshold) {
    // Check neighboring pixels
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int newX = x + dx;
            int newY = y + dy;

            // Check bounds
            if (newX >= 0 && newX < edges.cols &&
                newY >= 0 && newY < edges.rows &&
                edges.at<uchar>(newY, newX) == 0) {

                // If pixel is above low threshold, mark as edge
                if (nonMaxSuppressed.at<float>(newY, newX) > lowThreshold) {
                    edges.at<uchar>(newY, newX) = 255;
                    recursiveHysteresis(edges, nonMaxSuppressed, newX, newY, lowThreshold);
                }
            }
        }
    }
}