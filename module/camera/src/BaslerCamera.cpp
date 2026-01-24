#include "camera/BaslerCamera.hpp"

#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>

namespace sm
{
	BaslerCamera::BaslerCamera()
	{
		
	}

	BaslerCamera::~BaslerCamera()
	{
	}

	int BaslerCamera::Init() {
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

		return 0;
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