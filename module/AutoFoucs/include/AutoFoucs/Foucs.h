#include <opencv2/opencv.hpp>
//#include "AutoFoucs.h"
//#include<AutoFoucs/AutoFoucs.h>
// DLL 导出宏定义
#ifdef _WIN32
#ifdef AUTOFOUCS_EXPORTS           // 在编译 DLL 时定义
#define AUTOFOUCS_API __declspec(dllexport)
#else                              // 在使用 DLL 时使用
#define AUTOFOUCS_API __declspec(dllimport)
#endif
#else
#define AUTOFOUCS_API
#endif
AUTOFOUCS_API double ComputerTenengrad(cv::Mat& image);