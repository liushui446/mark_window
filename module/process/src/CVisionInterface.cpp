#include "process/CVisionInterface.hpp"
#include "camera/CameraManager.hpp"


namespace sm {
    
	/*
	=============== 图像接口Base类 ===============
	*/

	CVisionProcessBase::CVisionProcessBase()
	{

	}

	CVisionProcessBase::~CVisionProcessBase()
	{

	}

	APIErrCode CVisionProcessBase::Init()
	{

		return APIErrCode::SUCCESS;
	}

	APIErrCode CVisionProcessBase::CameraCapture(cv::Mat& img)
	{
		
		//相机控制
		//HardWare_Move::GetInstance().ControlRGBWHA_AllClose();

		// 获得一张图片
		int width = CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_WIDTH);
		int height = CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_HEIGHT);
		
		img = cv::Mat::zeros(height, width, CV_8UC1);
		CameraManager::GetInstance().GetCameraData(CameraID::CAMERA_ID_MAIN, img.data);

		return APIErrCode::SUCCESS;
	}
}