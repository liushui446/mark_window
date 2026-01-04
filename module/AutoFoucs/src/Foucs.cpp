#include "StdAfx.h"
#define AUTOFOUCS_EXPORTS  
#include<AutoFoucs/Foucs.h>
AUTOFOUCS_API double ComputerTenengrad(cv::Mat& image)
{
	cv::Mat gray_img, sobel_x, sobel_y, G;
	if (image.channels() == 3) {
		cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
	}
	//分别计算x/y方向梯度
	Sobel(gray_img, sobel_x, CV_32FC1, 1, 0);
	Sobel(gray_img, sobel_y, CV_32FC1, 0, 1);
	multiply(sobel_x, sobel_x, sobel_x);
	multiply(sobel_y, sobel_y, sobel_y);
	cv::Mat sqrt_mat = sobel_x + sobel_y;
	sqrt(sqrt_mat, G);
	cv::Mat mat_mean, mat_stddev;
	cv::meanStdDev(G, mat_mean, mat_stddev);
	double T = mat_mean.at<double>(0, 0) + 3 * mat_stddev.at<double>(0, 0);

	for (int row = 0; row < G.rows; ++row) {
		char* imgdata = G.ptr<char>(row);
		for (int col = 0; col < G.cols; ++col) {
			if (imgdata[col] <= T) {
				imgdata[col] = 0;
			}
		}
	}
	//cv::multiply(G, G, G);
	return mean(G)[0];
}