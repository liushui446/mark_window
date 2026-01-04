#pragma once
#ifndef FACET_MODEL_HPP
#define FACET_MODEL_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc.hpp>
class FacetEdgeDetector {
private:
    std::vector<cv::Mat> kernels;
    int windowSize;

    void initKernels();
    std::vector<float> computeBasisFunctions(int x, int y);
    //public:
    //    std::vector<cv::Mat> coefficients_s{ cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(),
    //                                      cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat() };
    //    std::vector<cv::Mat> coefficients_t{ cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(),
    //                                     cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat() };

public:
    explicit FacetEdgeDetector(int windowSize = 5);

    cv::Point2f computeSubpixelInPolar(
        float x, float y,
        float k2, float k3, float k4, float k5, float k6,
        float k7, float k8, float k9, float k10
    );
    std::vector<std::vector<cv::Mat>> allCoefficients;  // 每次调用都保存一份10个Mat
    cv::Mat detectEdges(const cv::Mat& image, float threshold = 30.0f);
    std::vector<cv::Point2f> getSubpixelEdgePoints(const cv::Mat& image, float threshold = 30.0f);
    cv::Mat visualizeEdgePoints(const cv::Mat& image, const std::vector<cv::Point2f>& edgePoints);
    void calculateGradientVectors(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY);
    void calculateGradientVectors(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY, std::vector<cv::Point>& edgePoints);
    cv::Mat gradientTemplateMatching(const cv::Mat& templateImage, const cv::Mat& targetImage, cv::Point& best_position);
    cv::Point2f gradientTemplateMatching(const cv::Mat& templateImage, const cv::Mat& targetImage, cv::Point& best_position, std::vector<cv::Point>& edgePoints, const cv::Mat& templateGradX, const cv::Mat& templateGradY);
    cv::Point2f gradientTemplateMatching2(const cv::Mat& templateImage, const cv::Mat& targetImage, cv::Point& best_position, std::vector<cv::Point>& edgePoints, const cv::Mat& templateGradX, const cv::Mat& templateGradY);
    cv::Point2f refinePosition(const cv::Mat& targetGradX, const cv::Mat& targetGradY,
        const cv::Mat& templateGradX, const cv::Mat& templateGradY, const std::vector<cv::Point>& edgePoints, cv::Point2f startPos, double range, double step);
    double computeGradientNormalizedSimilarity2(const  cv::Mat& gradX1, const  cv::Mat& gradY1,
        const  cv::Mat& gradX2, const  cv::Mat& gradY2, const std::vector<cv::Point>& edgePoints, double subx, double suby);
    cv::Point2f refinePosition1(const cv::Mat& targetGradX, const cv::Mat& targetGradY,
        const cv::Mat& templateGradX, const cv::Mat& templateGradY, const std::vector<cv::Point>& edgePoints, cv::Point2f startPos, double range, double step);
    cv::Point2f refinePosition2(const cv::Mat& targetGradX, const cv::Mat& targetGradY,
        const cv::Mat& templateGradX, const cv::Mat& templateGradY, const std::vector<cv::Point>& edgePoints, cv::Point2f startPos, double range, double step);
    void saveGradientVectors(const cv::Mat& image, const cv::Mat& gradX, const cv::Mat& gradY);
};

#endif // FACET_MODEL_HPP