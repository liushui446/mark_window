#pragma once
#include <QImage>
#include <opencv2/opencv.hpp>

QImage cvMat2QImage(const cv::Mat& mat);

cv::Mat QImage2cvMat(const QImage& image);