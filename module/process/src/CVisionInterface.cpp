#include "process/CVisionInterface.hpp"
#include "camera/CameraManager.hpp"


namespace sm {
    
	/*
	=============== 图像接口类 ===============
	*/

	CVisionInterface& CVisionInterface::Ins() {
		static CVisionInterface instance;
		return instance;
	}

	CVisionInterface::~CVisionInterface()
	{

	}


	CVisionInterface::CVisionInterface()
	{

	}

	APIErrCode CVisionInterface::Init()
	{
		return APIErrCode::SUCCESS;
	}

	APIErrCode CVisionInterface::CameraCapture(cv::Mat& img)
	{
		CameraID id = CameraID::CAMERA_ID_MAIN;
		//相机控制
		//HardWare_Move::GetInstance().ControlRGBWHA_AllClose();
		CameraManager::GetInstance().Init(id);

		// 获得一张图片
		int width = CameraManager::GetInstance().GetParaInt(id, CameraParameter::CAMERA_PARA_WIDTH);
		int height = CameraManager::GetInstance().GetParaInt(id, CameraParameter::CAMERA_PARA_HEIGHT);
		int channel = CameraManager::GetInstance().GetParaInt(id, CameraParameter::CAMERA_CHANNEL);
		
		img = cv::Mat::zeros(height, width, channel);
		CameraManager::GetInstance().GetCameraData(id, img.data);
		img.convertTo(img, CV_8UC3);

		return APIErrCode::SUCCESS;
	}
}