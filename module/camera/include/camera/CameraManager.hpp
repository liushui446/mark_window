#ifndef CAMERAMANAGER_H
#define CAMERAMANAGER_H

//���������ʵ��ͷ�ļ�

//core����
#include "camera/CCameraBase.hpp"

namespace sm {

	class Pimple
	{
	public:
		std::vector<CameraPtr> vecCameraList;//��Ų�ͬ���ʵ��
		std::map<string, CameraID> StringToID;//������һ����
		std::map<CameraID, CameraPtr> AccessByCameraID;

		//********** Basler *********//
	};

	class SM_EXPORTS CameraManager
	{
	public:
		static CameraManager& GetInstance();//����
		CameraManager();
		~CameraManager();

		int Init();

		int Test(CameraID id);

		//������
		int SoftWareTrigger(CameraID id);

		//�ȴ�һ��ͼ�ĵ���
		bool WaitForGrabOneImageFinish(CameraID id, int Getimagetime);

		//��ͼ
		int SaveImage(ImageType enSaveImageType, CameraID id);

		//��ȡ�������
		int GetParaInt(CameraID id, CameraParameter parm);

		int SetShootParams(CameraID id);

		//��ȡͼ������
		int GetCameraData(CameraID id, unsigned char* data);

		int UseCameraDemo();

		int StopGrabbing(CameraID id);

		int StartGrabbing(CameraID id);

		int CloseDevice(CameraID id);

		int SetExposureTime(CameraID id, double exposure);
		double GetExposureTime(CameraID id);
		int SetGain(CameraID id, double gain);
		double GetGain(CameraID id);
		//CameraIDת���ʵ������
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
		static shared_ptr<CameraManager> pIns;//����
	};

}
#endif