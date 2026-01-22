#ifndef BASLERCAMERA_HPP
#define BASLERCAMERA_HPP

#include "camera/CCameraBase.hpp"

namespace sm
{
	class BaslerCamera :public CCameraBase
	{
	public:
		BaslerCamera();

		//相机初始化
		int Init() override;

		//设置Pimple基本参数
		//int SetPimpleParameters();

		//设置相机属性参数
		//int SetParameters() override;

		//开始取流
		//int StartAcquisition();

		//软触发
		int SoftwareTrigger() override;

		//等图只有软触发的时候使用
		bool WaitForSingleImageCaptured(int GetImage_time) override;

		//获取图片数据
		int GetCameraData(unsigned char* data) override;

		//保存最近一张图片
		int SaveImage(ImageType imagetype) override;

		//关闭相机
		int CloseDevice() override;

	private:

	};
}
#endif
