#pragma once

//其他项目头文件
//#include <yaml-cpp/yaml.h>

#include "core/CommonCore.hpp"
#include "core/CameraCore.hpp"

namespace sm {
	//class CCameraBase;//ControlCameraBase

	class CCameraBase {
	public:

		//拍照阶段
		enum class ShootPhase {
			NoAction = 0,              // 没动作
			ReadyToShoot,              // 准备好进行拍照
			ShootComplete,             // 拍照完成
			ImageReached               // 图片到达
		};

		//获取图像方式
		enum class GetImageWay {
			CallBack = 0,              //回调函数
			GetImage                   //主动取流
		};

		//相机拍照模式
		enum class ShootMode
		{
			SoftwareTrigger = 0,        //软触发
			HardwareTrigger             //硬触发
		};

	public:

		CCameraBase() = default;
		virtual ~CCameraBase() {};

		//相机初始化函数
		virtual int Init() = 0;

		//深拷贝得到图像数据
		virtual int GetCameraData(unsigned char* data) = 0;

		//软触发
		virtual int SoftwareTrigger() = 0;

		//等待图像采集完成
		virtual bool WaitForSingleImageCaptured(int GetImage_time) = 0;

		//配置相机用户参数
		//virtual int ConfigureUserPara() = 0;

		//配置相机初始化参数
		//virtual int SetParameters() = 0;
		virtual int GetCameraParaInt(CameraParameter para) = 0;

		//关闭设备
		virtual int CloseDevice() = 0;

		//保存图像
		virtual int SaveImage(ImageType imagetype) = 0;

	};

	typedef shared_ptr<CCameraBase> CameraPtr;
}