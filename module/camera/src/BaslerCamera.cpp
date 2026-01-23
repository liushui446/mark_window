#include "camera/BaslerCamera.hpp"

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

	int BaslerCamera::SaveImage(ImageType imagetype)
	{
		return 0;
	}

	int BaslerCamera::CloseDevice()
	{
		return 0;
	}

}