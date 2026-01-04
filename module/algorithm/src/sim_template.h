#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>
void computeGradient(const cv::Mat& img, cv::Mat& gradX, cv::Mat& gradY, cv::Mat& magnitude, cv::Mat& direction);
double computeSimilarity(const cv::Mat& templateDir, const cv::Mat& imageDir, cv::Point matchPoint);
cv::Point templateMatching(const cv::Mat& image, const cv::Mat& templ);

void computeGradient(const cv::Mat& img, cv::Mat& gradX, cv::Mat& gradY);
double computeGradientNormalizedSimilarity(const cv::Mat& gradX1, const cv::Mat& gradY1,
    const cv::Mat& gradX2, const cv::Mat& gradY2);
double computeGradientNormalizedSimilarity1(const  cv::Mat& gradX1, const  cv::Mat& gradY1,
    const  cv::Mat& gradX2, const  cv::Mat& gradY2, const std::vector<cv::Point>& edgePoints);
cv::Point matchTemplateGradient(const cv::Mat& image, const cv::Mat& templateImg);
cv::Point matchTemplateGradient(const cv::Mat& image, const cv::Mat& gradX1, const cv::Mat& gradY1);
void extractEdgePointsWithGradient(const cv::Mat& img, std::vector<cv::Point>& edgePoints, std::vector<float>& gradientX, std::vector<float>& gradientY);

void saveGradientToImage(const  std::vector<cv::Point>& edgePoints, const  std::vector<float>& gradientX, const  std::vector<float>& gradientY, cv::Mat& gradImageX, cv::Mat& gradImageY);

void cropEdgeRegion(const cv::Mat& img, const cv::Mat& edgeImg, const std::vector<cv::Point>& edgePoints, cv::Mat& gradImageX, cv::Mat& gradImageY, cv::Mat& croppedGradX, cv::Mat& croppedGradY, cv::Mat& croppededge, cv::Mat& croppedImg);

cv::Point matchTemplateGradientOptimized(const cv::Mat& image, const cv::Mat& gradX1, const cv::Mat& gradY1, const std::vector<cv::Point>& edgePoints);

void drawTemplateEdges(cv::Mat& srcImage, std::vector<cv::Point>& edgePoints, cv::Point position);

double computeGradientNormalizedSimilarity0(const  cv::Mat& gradX1, const  cv::Mat& gradY1,
    const  cv::Mat& gradX2, const  cv::Mat& gradY2, const std::vector<cv::Point>& edgePoints, double subx, double suby);
