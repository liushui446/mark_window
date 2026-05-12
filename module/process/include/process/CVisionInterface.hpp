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
		//关闭相机
		APIErrCode CloseCamera();
		//相机拍照
		APIErrCode CameraCapture(cv::Mat& img);
		//停止采集
		APIErrCode StopCapture();
		//开始实时采集
		APIErrCode StartCapture();

	};

}
#endif//
