#ifndef CAMERAMANAGER_H
#define CAMERAMANAGER_H

//管理的相机实例头文件

//core定义
#include "camera/CCameraBase.hpp"

namespace sm {

	class Pimple
	{
	public:
		std::vector<CameraPtr> vecCameraList;//存放不同相机实例
		std::map<string, CameraID> StringToID;//留着万一有用
		std::map<CameraID, CameraPtr> AccessByCameraID;

		//********** Basler *********//
	};

	class SM_EXPORTS CameraManager
	{
	public:
		static CameraManager& GetInstance();//单例
		CameraManager();
		~CameraManager();

		int Init(CameraID id);

		//软触发
		int SoftWareTrigger(CameraID id);

		//等待一张图的到达
		bool WaitForGrabOneImageFinish(CameraID id, int Getimagetime);

		//存图
		int SaveImage(ImageType enSaveImageType, CameraID id);

		//获取相机参数
		int GetParaInt(CameraID id, CameraParameter parm);

		int SetShootParams(CameraID id);

		//获取图像数据
		int GetCameraData(CameraID id, unsigned char* data);

		int UseCameraDemo();

		//CameraID转相机实例对象
		inline CameraPtr IDtoCamPtr(CameraID id)
		{
			if (pMembers->AccessByCameraID.find(id) == pMembers->AccessByCameraID.end())
			{
				return nullptr;
			}
			else
			{
				return  pMembers->AccessByCameraID.find(id)->second;
			}
		}

	private:
		shared_ptr<Pimple> pMembers;
		static shared_ptr<CameraManager> pIns;//单例
	};

}
#endif