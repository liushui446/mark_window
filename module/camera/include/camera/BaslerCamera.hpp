#ifndef BASLERCAMERA_HPP
#define BASLERCAMERA_HPP

#include "camera/CCameraBase.hpp"

namespace sm
{
	//struct BaslerCamera::BaslerPimple
	//{
	//	CameraID id;
	//	std::string serialNumber; //序列号
	//	std::string szName; //相机自定义名称
	//	void* m_hDevHandle; //相机控制句柄

	//	std::atomic <uchar*> data; //图像数据指针(暂存)
	//};

	class BaslerCamera : public CCameraBase
	{
	public:
		BaslerCamera();
		~BaslerCamera() override;

		struct BaslerPimple
		{
			CameraID id;
			std::string serialNumber; //序列号
			std::string szName; //相机自定义名称
			void* m_hDevHandle; //相机控制句柄

			std::atomic <uchar*> data; //图像数据指针(暂存)
		};

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

		//回调函数
		//static void __stdcall CallbackStoreImage(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);

		//等图只有软触发的时候使用
		bool WaitForSingleImageCaptured(int GetImage_time) override;

		//获取图片数据
		int GetCameraData(unsigned char* data) override;

		//保存最近一张图片
		int SaveImage(ImageType imagetype) override;

		//关闭相机
		int CloseDevice() override;

	private:
		//struct BaslerPimple;
		shared_ptr<BaslerPimple> pBaslerMember;
	};
}
#endif
