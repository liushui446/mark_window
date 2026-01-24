#include "camera/BaslerCamera.hpp"

#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace sm
{
	BaslerCamera::BaslerCamera(CameraID id)
	{
		pBaslerMember = make_shared<BaslerPimple>();
		pBaslerMember->id = id;
		// 原子化赋值 nullptr
		pBaslerMember->data.store(nullptr, memory_order_relaxed);
		pBaslerMember->camerainfo.width = 0;
		pBaslerMember->camerainfo.height = 0;
		pBaslerMember->camerainfo.ImageSize = 0;

	}

	BaslerCamera::~BaslerCamera()
	{
	}

	int BaslerCamera::Init() {

		string img_path = "D:\\test_images\\123.jpg";
		Mat img = imread(img_path);
		if (img.empty()) {
			cerr << "读取图片失败！原因：" << endl;
			cerr << "1. 路径错误：" << img_path << endl;
			cerr << "2. 图片文件损坏/格式不支持" << endl;
			return -1;
		}

		// 2. 转换为 8位单通道（CV_8UC1）
		Mat img_8u;
		if (img.depth() != CV_8U || img.channels() != 1) {
			// 若原图片是彩色（3通道），先转灰度再转 8位；若已是单通道仅转深度
			if (img.channels() == 3) {
				cvtColor(img, img_8u, COLOR_BGR2GRAY); // 彩色转灰度
			}
			else {
				img.convertTo(img_8u, CV_8UC1); // 仅转换深度
			}
		}
		else {
			img_8u = img;
		}
		Mat img_continuous = img_8u.isContinuous() ? img_8u : img_8u.clone();
		
		// 4. 计算数据总字节数
		size_t img_size = img_continuous.total() * img_continuous.elemSize();
		// 5. 分配独立堆内存（替代直接赋值 Mat 的临时指针）
		uchar* new_data = new (nothrow) uchar[img_size]; // nothrow 避免分配失败抛异常
		if (new_data == nullptr) {
			cerr << "内存分配失败！需要 " << img_size << " 字节" << endl;
			return -1;
		}
		// 6. 拷贝 Mat 数据到独立内存
		memcpy_s(new_data, img_size, img_continuous.data, img_size);

		// ========== 原子更新指针（释放旧内存，避免泄漏） ==========
	// 7. 读取旧指针并释放
		uchar* old_data = pBaslerMember->data.load(memory_order_relaxed);
		if (old_data != nullptr) {
			delete[] old_data; // 释放第一次分配的内存
			old_data = nullptr;
		}
		// 8. 原子赋值新指针
		pBaslerMember->data.store(new_data, memory_order_relaxed);

		// ========== 更新相机信息 ==========
		pBaslerMember->camerainfo.width = img_continuous.cols;
		pBaslerMember->camerainfo.height = img_continuous.rows;
		pBaslerMember->camerainfo.ImageSize = img_size; // 直接用计算好的大小，避免重复计算
		pBaslerMember->camerainfo.channels = img_continuous.channels();

		cout << "图片读取并拷贝成功！数据长度：" << img_size << endl;

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
			memcpy_s(data, pBaslerMember->camerainfo.ImageSize, pBaslerMember->data, pBaslerMember->camerainfo.ImageSize);
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
		return 0;
	}

}