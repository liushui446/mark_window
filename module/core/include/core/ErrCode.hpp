#ifndef ERRCODE_HPP
#define ERRCODE_HPP

namespace sm
{
	enum class APIErrCode : unsigned int
	{
		// Common error code
		OK = 0x0000,
		SUCCESS = 0x0000,
		FAIL = 0xA001,
		ERROR_CAMERA_CAPTURE= 0xA005
	};
}

#endif
