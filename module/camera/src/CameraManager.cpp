#include "camera/CameraManager.hpp"
#include "camera/BaslerCamera.hpp"

#include <pylon/PylonIncludes.h>
#include <pylon/usb/PylonUsbIncludes.h>
#include <pylon/EnumParameterT.h>
#include <opencv2/opencv.hpp>
using namespace std;

using namespace Pylon;
using namespace GenApi;
using namespace cv;

namespace sm {

	shared_ptr<CameraManager> CameraManager::pIns(new CameraManager);

	CameraManager& CameraManager::GetInstance() {
		return *pIns;
	}

	CameraManager::CameraManager() {
		CameraID id = CameraID::CAMERA_ID_MAIN;
		pMembers = make_shared<Pimple>();
		auto cameraPtr = std::make_shared<BaslerCamera>(id);
		pMembers->AccessByCameraID.emplace(id, cameraPtr);
	}

	CameraManager::~CameraManager() {

	}

	int CameraManager::Init() {
		CameraID id = CameraID::CAMERA_ID_MAIN;
		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->Init();
		}
		else {
			return -1;
		}
		return 0;
	}

	int CameraManager::Test(CameraID id) {
		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->CameraTest();
		}
		else {
			return -1;
		}
		return 0;
	}

	int CameraManager::SoftWareTrigger(CameraID id) {
		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->SoftwareTrigger();
		}
		else {
			return -1;
		}
	}

	bool CameraManager::WaitForGrabOneImageFinish(CameraID id, int Getimagetime)
	{
		if (IDtoCamPtr(id) != nullptr)
		{
			return IDtoCamPtr(id)->WaitForSingleImageCaptured(Getimagetime);
		}
		else
		{
			return false;
		}
	}

	int CameraManager::SaveImage(ImageType enSaveImageType, CameraID id) {
		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->SaveImage(enSaveImageType);
		}
		else {
			return -1;
		}
	}

	int CameraManager::GetParaInt(CameraID id, CameraParameter parm) {
		if (IDtoCamPtr(id) != nullptr)
		{
			return IDtoCamPtr(id)->GetCameraParaInt(parm);
		}
		else
		{
			return -1;
		}
	}

	int CameraManager::SetShootParams(CameraID id)
	{
		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->SetShootParams();
		}
		else {
			return -1;
		}
	}

	int CameraManager::GetCameraData(CameraID id, unsigned char* data) {
		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->GetCameraData(data);
		}
		else {
			return -1;
		}
	}

	int CameraManager::UseCameraDemo()
	{
		PylonInitialize();

		try {
			CTlFactory& tlFactory = CTlFactory::GetInstance();

			ITransportLayer* pUsbTl = tlFactory.CreateTl("BaslerUsb");
			if (pUsbTl == nullptr) {
				cerr << "Cannot create USB transport!" << endl;
				PylonTerminate();
				return -1;
			}

			DeviceInfoList_t usbDevices;
			pUsbTl->EnumerateDevices(usbDevices);

			if (usbDevices.empty()) {
				cerr << "No USB camera found!" << endl;
				tlFactory.ReleaseTl(pUsbTl);
				PylonTerminate();
				return -1;
			}

			cout << "Found " << usbDevices.size() << " USB camera(s)" << endl;
			for (size_t i = 0; i < usbDevices.size(); ++i) {
				cout << "  Camera " << i + 1 << ": " << usbDevices[i].GetModelName()
					<< " S/N: " << usbDevices[i].GetSerialNumber() << endl;
			}

			CInstantCamera camera(tlFactory.CreateDevice(usbDevices[0]));
			cout << "Connected to: " << camera.GetDeviceInfo().GetModelName() << endl;

			camera.Open();

			INodeMap& nodeMap = camera.GetNodeMap();
			CIntegerParameter widthParam(nodeMap, "Width");
			if (widthParam.IsReadable()) {
				cout << "Width: " << widthParam.GetValue() << endl;
			}
			CIntegerParameter heightParam(nodeMap, "Height");
			if (heightParam.IsReadable()) {
				cout << "Height: " << heightParam.GetValue() << endl;
			}
			camera.StartGrabbing(GrabStrategy_LatestImageOnly);
			CImageFormatConverter formatConverter;
			formatConverter.OutputPixelFormat = PixelType_BGR8packed;
			CPylonImage pylonImage;
			cv::Mat opencvImage;

			const int framesToGrab = 100;
			for (int i = 0; i < framesToGrab && camera.IsGrabbing(); ++i)
			{
				CGrabResultPtr grabResult;
				camera.RetrieveResult(5000, grabResult, TimeoutHandling_ThrowException);

				if (grabResult->GrabSucceeded())
				{
					formatConverter.Convert(pylonImage, grabResult);
					opencvImage = cv::Mat(grabResult->GetHeight(), grabResult->GetWidth(),
						CV_8UC3, (uint8_t*)pylonImage.GetBuffer());
				}
				else
				{
					std::cerr << "Grab failed" << std::endl;
				}
			}

			camera.StopGrabbing();
			camera.Close();
			tlFactory.ReleaseTl(pUsbTl);
		}
		catch (const GenericException& e) {
			cerr << "USB camera exception: " << e.GetDescription() << endl;
			PylonTerminate();
			return -1;
		}

		PylonTerminate();
		cout << "Camera demo finished" << endl;
		return 0;
	}

	int CameraManager::StopGrabbing(CameraID id)
	{
		auto cam = IDtoCamPtr(id);
		if (cam) {
			return cam->StopGrabbing();
		}
		return -1;
	}

	int CameraManager::StartGrabbing(CameraID id)
	{
		auto cam = IDtoCamPtr(id);
		if (cam) {
			return cam->StartGrabbing();
		}
		return -1;
	}

	int CameraManager::CloseDevice(CameraID id) {
		auto cam = IDtoCamPtr(id);
		if (cam) {
			return cam->CloseDevice();
		}
		return -1;
	}

	int CameraManager::SetExposureTime(CameraID id, double exposure) {
		auto cam = IDtoCamPtr(id);
		if (cam) {
			return cam->SetExposureTime(exposure);
		}
		return -1;
	}

	double CameraManager::GetExposureTime(CameraID id) {
		auto cam = IDtoCamPtr(id);
		if (cam) {
			return cam->GetExposureTime();
		}
		return 0.0;
	}

	int CameraManager::SetGain(CameraID id, double gain) {
		auto cam = IDtoCamPtr(id);
		if (cam) {
			return cam->SetGain(gain);
		}
		return -1;
	}

	double CameraManager::GetGain(CameraID id) {
		auto cam = IDtoCamPtr(id);
		if (cam) {
			return cam->GetGain();
		}
		return 0.0;
	}
}