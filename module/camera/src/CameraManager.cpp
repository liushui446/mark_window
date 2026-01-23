#include "camera/CameraManager.hpp"

namespace sm {

	shared_ptr<CameraManager> CameraManager::pIns(new CameraManager);

	CameraManager& CameraManager::GetInstance() {
		return *pIns;
	}

	CameraManager::CameraManager() {
		
	}

	CameraManager::~CameraManager() {
		
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

	int CameraManager::GetCameraData(CameraID id, unsigned char* data) {
		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->GetCameraData(data);
		}
		else {
			return -1;
		}
	}

}