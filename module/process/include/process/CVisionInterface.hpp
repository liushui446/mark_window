#ifndef CVISONINTERFACE_HPP
#define CVISONINTERFACE_HPP

#include"core/CommonCore.hpp"

// Include files to use OpenCV API.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

namespace sm {
    
	//********************** 图像接口Base类 **********************
	class SM_EXPORTS CVisionInterface
	{
	public:
		static CVisionInterface& Ins();
		~CVisionInterface();

	public:
		CVisionInterface(const CVisionInterface&) = delete;
		CVisionInterface(CVisionInterface&&) = delete;
		CVisionInterface& operator=(const CVisionInterface&) = delete;
		CVisionInterface& operator=(CVisionInterface&&) = delete;

		CVisionInterface();

	public:
		// 初始化
		APIErrCode Init();
		//相机拍照
		APIErrCode CameraCapture(cv::Mat& img);

	};

}
#endif//
