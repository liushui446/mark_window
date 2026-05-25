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
		pBaslerMember->data.store(nullptr, memory_order_relaxed);
		pBaslerMember->camerainfo.width = 0;
		pBaslerMember->camerainfo.height = 0;
		pBaslerMember->camerainfo.ImageSize = 0;
		m_grabbing = false;

	}

	BaslerCamera::~BaslerCamera()
	{
		if (m_grabThread.joinable()) {
			m_grabbing = false;
			m_grabThread.join();
		}
		if (m_camera && m_camera->IsOpen()) {
			if (m_camera->IsGrabbing())
				m_camera->StopGrabbing();
			m_camera->Close();
		}
	}

	int BaslerCamera::Init() {
		try {
			{
				lock_guard<mutex> lock(g_pylonInitMutex);
				if (!g_pylonInitialized) {
					PylonInitialize();
					g_pylonInitialized = true;
				}
			}

			CTlFactory& tlFactory = CTlFactory::GetInstance();
			ITransportLayer* pUsbTl = tlFactory.CreateTl("BaslerUsb");
			if (!pUsbTl) {
				cerr << "Cannot create USB transport!" << endl;
				return -1;
			}

			DeviceInfoList_t usbDevices;
			pUsbTl->EnumerateDevices(usbDevices);
			if (usbDevices.empty()) {
				cerr << "No USB camera found!" << endl;
				tlFactory.ReleaseTl(pUsbTl);
				return -1;
			}

			cout << "Found " << usbDevices.size() << " USB camera(s)" << endl;
			for (size_t i = 0; i < usbDevices.size(); ++i) {
				cout << "  Camera " << i + 1 << ": " << usbDevices[i].GetModelName()
					<< " Serial: " << usbDevices[i].GetSerialNumber() << endl;
			}

			m_camera.reset(new CBaslerUsbInstantCamera(tlFactory.CreateDevice(usbDevices[0])));
			m_camera->Open();
			tlFactory.ReleaseTl(pUsbTl);

			INodeMap& nodeMap = m_camera->GetNodeMap();
			CFloatPtr exposureTime = nodeMap.GetNode("ExposureTime");
			if (IsWritable(exposureTime))
				exposureTime->SetValue(4000.0);
			CFloatPtr gain = nodeMap.GetNode("Gain");
			if (IsWritable(gain))
				gain->SetValue(3.0);

			m_converter.reset(new CImageFormatConverter);
			m_converter->OutputPixelFormat = PixelType_BGR8packed;
			m_pylonImage.reset(new CPylonImage);

			CIntegerParameter widthParam(nodeMap, "Width");
			CIntegerParameter heightParam(nodeMap, "Height");
			pBaslerMember->camerainfo.width = widthParam.GetValue();
			pBaslerMember->camerainfo.height = heightParam.GetValue();
			pBaslerMember->camerainfo.channels = 3;
			pBaslerMember->camerainfo.ImageSize = pBaslerMember->camerainfo.width *
				pBaslerMember->camerainfo.height * 3;

			m_camera->StartGrabbing(GrabStrategy_LatestImageOnly);
			m_grabbing = true;
			m_grabThread = thread(&BaslerCamera::GrabLoop, this);

			cout << "Camera init success, resolution: " << pBaslerMember->camerainfo.width
				<< " x " << pBaslerMember->camerainfo.height << endl;
			return 0;
		}
		catch (const GenericException& e) {
			cerr << "BaslerCamera::Init exception: " << e.GetDescription() << endl;
			return -1;
		}
	}

	void BaslerCamera::GrabLoop()
	{
		while (m_grabbing) {
			CGrabResultPtr grabResult;
			if (m_camera->RetrieveResult(5000, grabResult, TimeoutHandling_ThrowException)) {
				if (grabResult->GrabSucceeded()) {
					m_converter->Convert(*m_pylonImage, grabResult);
					cv::Mat tmp(grabResult->GetHeight(), grabResult->GetWidth(),
						CV_8UC3, (uint8_t*)m_pylonImage->GetBuffer());
					{
						lock_guard<mutex> lock(m_frameMutex);
						tmp.copyTo(m_frame);
					}
				}
			}
		}
	}

	int BaslerCamera::CameraTest() {
		cv::Mat currentFrame;
		{
			lock_guard<mutex> lock(m_frameMutex);
			if (m_frame.empty()) {
				cerr << "No frame captured yet" << endl;
				return -1;
			}
			currentFrame = m_frame.clone();
		}

		if (!currentFrame.isContinuous())
			currentFrame = currentFrame.clone();

		size_t imgSize = currentFrame.total() * currentFrame.elemSize();
		uchar* newData = new (nothrow) uchar[imgSize];
		if (!newData) {
			cerr << "Memory allocation failed" << endl;
			return -1;
		}
		memcpy(newData, currentFrame.data, imgSize);

		uchar* oldData = pBaslerMember->data.exchange(newData);
		if (oldData)
			delete[] oldData;

		pBaslerMember->camerainfo.width = currentFrame.cols;
		pBaslerMember->camerainfo.height = currentFrame.rows;
		pBaslerMember->camerainfo.channels = currentFrame.channels();
		pBaslerMember->camerainfo.ImageSize = imgSize;

		cout << "Captured frame, size: " << imgSize << " bytes" << endl;
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
			uchar* pData = pBaslerMember->data.load(std::memory_order_acquire);
			if (pData == nullptr) {
				std::cerr << "Image data is empty" << std::endl;
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
		if (m_grabThread.joinable()) {
			m_grabbing = false;
			m_grabThread.join();
		}

		if (m_camera && m_camera->IsOpen()) {
			if (m_camera->IsGrabbing()) {
				m_camera->StopGrabbing();
			}
			m_camera->Close();
		}

		std::cout << "Camera device closed" << std::endl;
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
			cerr << "Camera not open, cannot start grabbing" << endl;
			return -1;
		}
		if (m_camera->IsGrabbing()) {
			return 0;
		}
		m_camera->StartGrabbing(GrabStrategy_LatestImageOnly);
		return 0;
	}

	int BaslerCamera::SetExposureTime(double exposure)
	{
		if (!m_camera || !m_camera->IsOpen()) return -1;
		try {
			INodeMap& nodeMap = m_camera->GetNodeMap();
			CFloatPtr exposureNode = nodeMap.GetNode("ExposureTime");
			if (IsWritable(exposureNode)) {
				exposureNode->SetValue(exposure);
				return 0;
			}
		}
		catch (const GenericException& e) {
			cerr << "SetExposureTime failed: " << e.GetDescription() << endl;
		}
		return -1;
	}

	double BaslerCamera::GetExposureTime()
	{
		if (!m_camera || !m_camera->IsOpen()) return 0.0;
		try {
			INodeMap& nodeMap = m_camera->GetNodeMap();
			CFloatPtr exposureNode = nodeMap.GetNode("ExposureTime");
			if (IsReadable(exposureNode))
				return exposureNode->GetValue();
		}
		catch (const GenericException& e) {
			cerr << "GetExposureTime failed: " << e.GetDescription() << endl;
		}
		return 0.0;
	}

	int BaslerCamera::SetGain(double gain)
	{
		if (!m_camera || !m_camera->IsOpen()) return -1;
		try {
			INodeMap& nodeMap = m_camera->GetNodeMap();
			CFloatPtr gainNode = nodeMap.GetNode("Gain");
			if (IsWritable(gainNode)) {
				gainNode->SetValue(gain);
				return 0;
			}
		}
		catch (const GenericException& e) {
			cerr << "SetGain failed: " << e.GetDescription() << endl;
		}
		return -1;
	}

	double BaslerCamera::GetGain()
	{
		if (!m_camera || !m_camera->IsOpen()) return 0.0;
		try {
			INodeMap& nodeMap = m_camera->GetNodeMap();
			CFloatPtr gainNode = nodeMap.GetNode("Gain");
			if (IsReadable(gainNode))
				return gainNode->GetValue();
		}
		catch (const GenericException& e) {
			cerr << "GetGain failed: " << e.GetDescription() << endl;
		}
		return 0.0;
	}
}