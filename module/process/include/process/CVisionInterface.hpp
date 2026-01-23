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
	class SM_EXPORTS CVisionProcessBase
	{
	protected:
		CVisionProcessBase();
		~CVisionProcessBase();

	public:
		CVisionProcessBase(const CVisionProcessBase&) = delete;
		CVisionProcessBase(CVisionProcessBase&&) = delete;
		CVisionProcessBase& operator=(const CVisionProcessBase&) = delete;
		CVisionProcessBase& operator=(CVisionProcessBase&&) = delete;

	public:
		// 初始化
		APIErrCode Init();
		//相机拍照
		APIErrCode CameraCapture(cv::Mat& img);

	};

}
#endif//
