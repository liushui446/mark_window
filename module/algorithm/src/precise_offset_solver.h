
#ifndef PRECISE_OFFSET_SOLVER_H
#define PRECISE_OFFSET_SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

class precise_offset_solver
{
public:
    // 使用图像和点集进行旋转和平移的拟合优化，返回0表示成功
    static int findPreciseOffset(const cv::Mat& image, std::vector<cv::Point2f>& points, double& r, double& x, double& y);
    static int findRigidTransform2D(const std::vector<cv::Point2f>& modelPoints,
        const std::vector<cv::Point2f>& dataPoints,
        double& theta, double& dx, double& dy,
        int max_iter , double epsilon );


private:
    static int calcJacobian(cv::Mat& jac, const cv::Mat& image, std::vector<cv::Point2f>& points, double r, double x, double y);
    static int calcError(cv::Mat& err, const cv::Mat& image, std::vector<cv::Point2f>& points, double r, double x, double y);
    static void calcDeriv(const cv::Mat& err1, const cv::Mat& err2, double h, cv::Mat& res);
};

#endif // PRECISE_OFFSET_SOLVER_H
#pragma once
