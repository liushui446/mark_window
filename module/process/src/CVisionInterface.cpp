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
		//相机初始化
		CameraManager::GetInstance().Init();
		return APIErrCode::SUCCESS;
	}

	APIErrCode CVisionInterface::CloseCamera()
	{
		CameraID id = CameraID::CAMERA_ID_MAIN;

		// 1. 先停止实时采集流
		CameraManager::GetInstance().StopGrabbing(id);

		// 2. 彻底关闭相机设备硬件资源
		int ret = CameraManager::GetInstance().CloseDevice(id);

		if (ret != 0) {
			// 这里复用了现有的错误码，你也可以在 CommonCore.hpp 中定义专门的 ERROR_CAMERA_CLOSE
			return APIErrCode::ERROR_CAMERA_CAPTURE;
		}

		return APIErrCode::SUCCESS;
	}

	APIErrCode CVisionInterface::CameraCapture(cv::Mat& img)
	{
		CameraID id = CameraID::CAMERA_ID_MAIN;

		//相机控制
		//HardWare_Move::GetInstance().ControlRGBWHA_AllClose();
		CameraManager::GetInstance().Test(id);

		// 获得一张图片
		int width = CameraManager::GetInstance().GetParaInt(id, CameraParameter::CAMERA_PARA_WIDTH);
		int height = CameraManager::GetInstance().GetParaInt(id, CameraParameter::CAMERA_PARA_HEIGHT);
		int channel = CameraManager::GetInstance().GetParaInt(id, CameraParameter::CAMERA_CHANNEL);
		int type = (channel == 1) ? CV_8UC1 : CV_8UC3;
		img = cv::Mat(height, width, type);
		if (CameraManager::GetInstance().GetCameraData(id, img.data) != 0) {
			return APIErrCode::ERROR_CAMERA_CAPTURE;
		}
		//img.convertTo(img, CV_8UC3);

		return APIErrCode::SUCCESS;
	}

	APIErrCode CVisionInterface::StartCapture()
	{
		CameraManager::GetInstance().StartGrabbing(CameraID::CAMERA_ID_MAIN);
		return APIErrCode::SUCCESS;
	}

	APIErrCode CVisionInterface::StopCapture()
	{
		CameraManager::GetInstance().StopGrabbing(CameraID::CAMERA_ID_MAIN);
		return APIErrCode::SUCCESS;
	}
}