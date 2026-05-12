#include "camera/BaslerCamera.hpp"

#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>
#include <pylon/usb/PylonUsbIncludes.h>
#include <pylon/EnumParameterT.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Pylon;
using namespace GenApi;

namespace sm
{
	static bool g_pylonInitialized = false;
	static mutex g_pylonInitMutex;
	BaslerCamera::BaslerCamera(CameraID id)
	{
		pBaslerMember = make_shared<BaslerPimple>();
		pBaslerMember->id = id;
		// 原子化赋值 nullptr
		pBaslerMember->data.store(nullptr, memory_order_relaxed);
		pBaslerMember->camerainfo.width = 0;
		pBaslerMember->camerainfo.height = 0;
		pBaslerMember->camerainfo.ImageSize = 0;
		m_grabbing = false;

	}

	BaslerCamera::~BaslerCamera()
	{
		// 停止抓图线程
		if (m_grabThread.joinable()) {
			m_grabbing = false;
			m_grabThread.join();
		}
		// 关闭相机
		if (m_camera && m_camera->IsOpen()) {
			if (m_camera->IsGrabbing())
				m_camera->StopGrabbing();
			m_camera->Close();
		}
	}

	int BaslerCamera::Init() {

		//打开相机
		//OpenCamera();

		//return 0;

		try {
			// 1. 确保 Pylon 运行时只初始化一次
			{
				lock_guard<mutex> lock(g_pylonInitMutex);
				if (!g_pylonInitialized) {
					PylonInitialize();
					g_pylonInitialized = true;
				}
			}

			// 2. 获取传输层工厂，只枚举 USB 相机
			CTlFactory& tlFactory = CTlFactory::GetInstance();
			ITransportLayer* pUsbTl = tlFactory.CreateTl("BaslerUsb");
			if (!pUsbTl) {
				cerr << "无法加载 USB 传输层！" << endl;
				return -1;
			}

			DeviceInfoList_t usbDevices;
			pUsbTl->EnumerateDevices(usbDevices);
			if (usbDevices.empty()) {
				cerr << "未找到任何 USB 相机！" << endl;
				tlFactory.ReleaseTl(pUsbTl);
				return -1;
			}

			// 打印找到的相机信息
			cout << "找到 " << usbDevices.size() << " 个 USB 相机：" << endl;
			for (size_t i = 0; i < usbDevices.size(); ++i) {
				cout << "  相机 " << i + 1 << ": " << usbDevices[i].GetModelName()
					<< " 序列号: " << usbDevices[i].GetSerialNumber() << endl;
			}

			// 3. 创建相机对象并打开
			m_camera.reset(new CBaslerUsbInstantCamera(tlFactory.CreateDevice(usbDevices[0])));
			m_camera->Open();
			tlFactory.ReleaseTl(pUsbTl);

			// 4. 设置相机参数（可选）
			INodeMap& nodeMap = m_camera->GetNodeMap();
			CFloatPtr exposureTime = nodeMap.GetNode("ExposureTime");
			if (IsWritable(exposureTime))
				exposureTime->SetValue(4000.0);   // 40ms
			CFloatPtr gain = nodeMap.GetNode("Gain");
			if (IsWritable(gain))
				gain->SetValue(3.0);

			// 5. 初始化格式转换器（输出 BGR 三通道）
			m_converter.reset(new CImageFormatConverter);
			m_converter->OutputPixelFormat = PixelType_BGR8packed;
			m_pylonImage.reset(new CPylonImage);

			// 6. 获取图像尺寸并更新到 camerainfo
			CIntegerParameter widthParam(nodeMap, "Width");
			CIntegerParameter heightParam(nodeMap, "Height");
			pBaslerMember->camerainfo.width = widthParam.GetValue();
			pBaslerMember->camerainfo.height = heightParam.GetValue();
			pBaslerMember->camerainfo.channels = 3;   // BGR 是三通道
			pBaslerMember->camerainfo.ImageSize = pBaslerMember->camerainfo.width *
				pBaslerMember->camerainfo.height * 3;

			// 7. 开始连续抓取（策略：只保留最新帧）
			m_camera->StartGrabbing(GrabStrategy_LatestImageOnly);
			m_grabbing = true;
			m_grabThread = thread(&BaslerCamera::GrabLoop, this);

			cout << "相机初始化成功，分辨率：" << pBaslerMember->camerainfo.width
				<< " x " << pBaslerMember->camerainfo.height << endl;
			return 0;
		}
		catch (const GenericException& e) {
			cerr << "BaslerCamera::Init 异常: " << e.GetDescription() << endl;
			return -1;
		}
	}
	void BaslerCamera::GrabLoop()
	{
		while (m_grabbing) {
			CGrabResultPtr grabResult;
			// 等待一帧，超时 1000ms
			if (m_camera->RetrieveResult(5000, grabResult, TimeoutHandling_ThrowException)) {
				if (grabResult->GrabSucceeded()) {
					// 转换为 BGR 格式
					m_converter->Convert(*m_pylonImage, grabResult);
					cv::Mat tmp(grabResult->GetHeight(), grabResult->GetWidth(),
						CV_8UC3, (uint8_t*)m_pylonImage->GetBuffer());
					// 加锁拷贝最新帧
					{
						lock_guard<mutex> lock(m_frameMutex);
						tmp.copyTo(m_frame);
					}
				}
			}
		}
	}
	int BaslerCamera::CameraTest() {

		//string img_path = "D:\\test_images\\123.jpg";
		//Mat img = imread(img_path);
		//if (img.empty()) {
		//	cerr << "读取图片失败！原因：" << endl;
		//	cerr << "1. 路径错误：" << img_path << endl;
		//	cerr << "2. 图片文件损坏/格式不支持" << endl;
		//	return -1;
		//}

		//// 2. 转换为 8位单通道（CV_8UC1）
		//Mat img_8u;
		//if (img.depth() != CV_8U || img.channels() != 1) {
		//	// 若原图片是彩色（3通道），先转灰度再转 8位；若已是单通道仅转深度
		//	if (img.channels() == 3) {
		//		cvtColor(img, img_8u, COLOR_BGR2GRAY); // 彩色转灰度
		//	}
		//	else {
		//		img.convertTo(img_8u, CV_8UC1); // 仅转换深度
		//	}
		//}
		//else {
		//	img_8u = img;
		//}
		//Mat img_continuous = img_8u.isContinuous() ? img_8u : img_8u.clone();

		//// 4. 计算数据总字节数
		//size_t img_size = img_continuous.total() * img_continuous.elemSize();
		//// 5. 分配独立堆内存（替代直接赋值 Mat 的临时指针）
		//uchar* new_data = new (nothrow) uchar[img_size]; // nothrow 避免分配失败抛异常
		//if (new_data == nullptr) {
		//	cerr << "内存分配失败！需要 " << img_size << " 字节" << endl;
		//	return -1;
		//}
		//// 6. 拷贝 Mat 数据到独立内存
		//memcpy_s(new_data, img_size, img_continuous.data, img_size);

		//// ========== 原子更新指针（释放旧内存，避免泄漏） ==========
		//// 7. 读取旧指针并释放
		//uchar* old_data = pBaslerMember->data.load(memory_order_relaxed);
		//if (old_data != nullptr) {
		//	delete[] old_data; // 释放第一次分配的内存
		//	old_data = nullptr;
		//}
		//// 8. 原子赋值新指针
		//pBaslerMember->data.store(new_data, memory_order_relaxed);

		//// ========== 更新相机信息 ==========
		//pBaslerMember->camerainfo.width = img_continuous.cols;
		//pBaslerMember->camerainfo.height = img_continuous.rows;
		//pBaslerMember->camerainfo.ImageSize = img_size; // 直接用计算好的大小，避免重复计算
		//pBaslerMember->camerainfo.channels = img_continuous.channels();

		//cout << "图片读取并拷贝成功！数据长度：" << img_size << endl;
		//return 0;
		// 从后台线程的最新帧中获取一帧
		cv::Mat currentFrame;
		{
			lock_guard<mutex> lock(m_frameMutex);
			if (m_frame.empty()) {
				cerr << "尚未获取到任何图像帧" << endl;
				return -1;
			}
			currentFrame = m_frame.clone();
		}

		// 确保图像数据连续
		if (!currentFrame.isContinuous())
			currentFrame = currentFrame.clone();

		size_t imgSize = currentFrame.total() * currentFrame.elemSize();
		uchar* newData = new (nothrow) uchar[imgSize];
		if (!newData) {
			cerr << "内存分配失败，需要 " << imgSize << " 字节" << endl;
			return -1;
		}
		memcpy(newData, currentFrame.data, imgSize);

		// 原子替换 pBaslerMember->data 中的指针，并释放旧内存
		uchar* oldData = pBaslerMember->data.exchange(newData);
		if (oldData)
			delete[] oldData;

		// 更新相机信息（宽高通道等，应与当前帧一致）
		pBaslerMember->camerainfo.width = currentFrame.cols;
		pBaslerMember->camerainfo.height = currentFrame.rows;
		pBaslerMember->camerainfo.channels = currentFrame.channels();
		pBaslerMember->camerainfo.ImageSize = imgSize;

		cout << "实时采集一帧成功，大小: " << imgSize << " 字节" << endl;
		return 0;
	}

	int BaslerCamera::OpenCamera() {
		if (pBaslerMember->m_hDevHandle == nullptr) {
			return -1;
		}

		return 0;
	}

	int BaslerCamera::SoftwareTrigger()
	{
		return 0;
	}

	bool BaslerCamera::WaitForSingleImageCaptured(int GetImage_time)
	{
		
		return true;
	}

	int BaslerCamera::GetCameraData(unsigned char* data)
	{
		if (pBaslerMember == nullptr)
		{
			return -1;
		}
		else
		{
			//uchar* pData = pBaslerMember->data.load(std::memory_order_relaxed); // 若为 atomic<uchar*>
			//// 非原子指针：uchar* pData = pBaslerMember->data;
			//if (pData == nullptr) {
			//	std::cerr << "错误：pBaslerMember->data 是 nullptr！" << std::endl;
			//	return 0;
			//}
			//memcpy_s(data, pBaslerMember->camerainfo.ImageSize, pBaslerMember->data, pBaslerMember->camerainfo.ImageSize);
			uchar* pData = pBaslerMember->data.load(std::memory_order_acquire);
			if (pData == nullptr) {
				std::cerr << "错误：图像数据为空" << std::endl;
				return -1;
			}
			memcpy_s(data, pBaslerMember->camerainfo.ImageSize, pData, pBaslerMember->camerainfo.ImageSize);
		}
		return 0;
	}

	int BaslerCamera::SetShootParams()
	{
		int nRet = true;
		size_t img_width;
		size_t img_height;
		
		pBaslerMember->camerainfo.width = img_width;
		pBaslerMember->camerainfo.height = img_height;
		pBaslerMember->camerainfo.ImageSize = pBaslerMember->camerainfo.width * pBaslerMember->camerainfo.height * sizeof(uchar);

		return nRet;
	}

	int BaslerCamera::GetCameraParaInt(CameraParameter para) {
		switch (para) {
		case CameraParameter::CAMERA_PARA_HEIGHT:
			return pBaslerMember->camerainfo.height;
		case CameraParameter::CAMERA_PARA_WIDTH:
			return pBaslerMember->camerainfo.width;
		case CameraParameter::CAMERA_PARA_SIZE:
			return pBaslerMember->camerainfo.ImageSize;
		case CameraParameter::CAMERA_CHANNEL:
			return pBaslerMember->camerainfo.channels;
		default:
			return 0;
		}
	}

	int BaslerCamera::SaveImage(ImageType imagetype)
	{
		return 0;
	}

	int BaslerCamera::CloseDevice()
	{
		// 1. 停止后台抓图线程
		if (m_grabThread.joinable()) {
			m_grabbing = false;
			m_grabThread.join();
		}

		// 2. 停止抓图并关闭相机句柄
		if (m_camera && m_camera->IsOpen()) {
			if (m_camera->IsGrabbing()) {
				m_camera->StopGrabbing();
			}
			m_camera->Close();
		}

		std::cout << "相机设备已关闭" << std::endl;
		return 0;
	}

	int BaslerCamera::StopGrabbing()
	{
		if (m_camera && m_camera->IsGrabbing()) {
			m_camera->StopGrabbing();
		}
		return 0;
	}

	int BaslerCamera::StartGrabbing()
	{
		if (!m_camera || !m_camera->IsOpen()) {
			cerr << "相机未打开，无法开始抓图" << endl;
			return -1;
		}
		// 如果已经抓图中，先停止（可选）
		if (m_camera->IsGrabbing()) {
			// 可以不做处理，或者先停止再开始
			return 0;
		}
		m_camera->StartGrabbing(GrabStrategy_LatestImageOnly);  // 或者 OneByOne 根据你的配置
		// 如果之前有后台线程，需要重新启动（如果你用后台线程）
		// m_grabbing = true;
		// m_grabThread = thread(&BaslerCamera::GrabLoop, this);
		return 0;
	}
}