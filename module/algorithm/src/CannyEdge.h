// CannyEdgeDetection.h
#ifndef CANNY_EDGE_H
#define CANNY_EDGE_H
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

class CannyEdge {
public:
    static cv::Mat detectEdges(const cv::Mat& inputImage, std::vector<cv::Point>& edgePoints, int&binarize_thresh);
    static cv::Mat detectEdges(const cv::Mat& inputImage, std::vector<cv::Point>& edgePoints);
private:
    static cv::Mat applyGaussianFilter(const cv::Mat& image);
    static void computeSobelGradients(const cv::Mat& image, cv::Mat& gradientMagnitude, cv::Mat& gradientX, cv::Mat& gradientY);
    static cv::Mat nonMaximumSuppression(const cv::Mat& gradientMagnitude, const cv::Mat& gradientX, const cv::Mat& gradientY);
    static cv::Mat hysteresisThresholding(const cv::Mat& nonMaxSuppressed, double lowThreshold, double highThreshold);
    static void recursiveHysteresis(cv::Mat& edges, const cv::Mat& nonMaxSuppressed, int x, int y, double lowThreshold);
};

#endif // CANNY_EDGE_DETECTION_H