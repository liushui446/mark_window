#ifndef BASLERCAMERA_HPP
#define BASLERCAMERA_HPP

#include "camera/CCameraBase.hpp"
namespace Pylon {
	class CBaslerUsbInstantCamera;
	class CImageFormatConverter;
	class CPylonImage;
}
#include <atomic>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
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
		BaslerCamera(CameraID id);
		~BaslerCamera() override;

		struct BaslerPimple
		{
			CameraID id;
			std::string serialNumber; //序列号
			std::string szName; //相机自定义名称
			void* m_hDevHandle; //相机控制句柄
			CameraConfig camerainfo;

			std::atomic <uchar*> data; //图像数据指针(暂存)

			BaslerPimple()
			{
				id = CameraID::CAMERA_ID_NONE;
				serialNumber = "";
				szName = "";
				m_hDevHandle = nullptr;
				data = nullptr;
			}
			~BaslerPimple() {};
		};

		//相机初始化
		int Init() override;

		//测试接口
		int CameraTest() override;

		//打开相机
		int OpenCamera();

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

		//获取相机参数
		int GetCameraParaInt(CameraParameter para);

		//获取图片数据
		int GetCameraData(unsigned char* data) override;

		//设置捕获图像参数
		int SetShootParams();

		//保存最近一张图片
		int SaveImage(ImageType imagetype) override;

		//关闭相机
		int CloseDevice() override;
		
		//停止实时采集
		int StopGrabbing() override;

		//开始实时采集
		int StartGrabbing() override;   // 新增
	private:
		// 连续抓图的后台线程函数
		void GrabLoop();

		// Pylon 相机对象（使用智能指针避免头文件依赖）
		std::unique_ptr<Pylon::CBaslerUsbInstantCamera> m_camera;
		std::unique_ptr<Pylon::CImageFormatConverter> m_converter;
		std::unique_ptr<Pylon::CPylonImage> m_pylonImage;

		cv::Mat m_frame;              // 最新一帧（BGR格式）
		std::mutex m_frameMutex;      // 保护 m_frame
		std::thread m_grabThread;     // 后台抓图线程
		std::atomic<bool> m_grabbing; // 控制抓图线程运行
		//struct BaslerPimple;
		shared_ptr<BaslerPimple> pBaslerMember;
	};
}
#endif
