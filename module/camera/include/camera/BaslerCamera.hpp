#ifndef BASLERCAMERA_HPP
#define BASLERCAMERA_HPP

#include "camera/CCameraBase.hpp"
namespace Pylon {
	class CBaslerUsbInstantCamera;
	class CImageFormatConverter;
	class CPylonImage;
}
#include <atomic>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
namespace sm
{
	//struct BaslerCamera::BaslerPimple
	//{
	//	CameraID id;
	//	std::string serialNumber; //���к�
	//	std::string szName; //����Զ�������
	//	void* m_hDevHandle; //������ƾ��

	//	std::atomic <uchar*> data; //ͼ������ָ��(�ݴ�)
	//};

	class BaslerCamera : public CCameraBase
	{
	public:
		BaslerCamera(CameraID id);
		~BaslerCamera() override;

		struct BaslerPimple
		{
			CameraID id;
			std::string serialNumber; //���к�
			std::string szName; //����Զ�������
			void* m_hDevHandle; //������ƾ��
			CameraConfig camerainfo;

			std::atomic <uchar*> data; //ͼ������ָ��(�ݴ�)

			BaslerPimple()
			{
				id = CameraID::CAMERA_ID_NONE;
				serialNumber = "";
				szName = "";
				m_hDevHandle = nullptr;
				data = nullptr;
			}
			~BaslerPimple() {};
		};

		//�����ʼ��
		int Init() override;

		//���Խӿ�
		int CameraTest() override;

		//�����
		int OpenCamera();

		//����Pimple��������
		//int SetPimpleParameters();

		//����������Բ���
		//int SetParameters() override;

		//��ʼȡ��
		//int StartAcquisition();

		//������
		int SoftwareTrigger() override;

		//�ص�����
		//static void __stdcall CallbackStoreImage(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);

		//��ͼֻ����������ʱ��ʹ��
		bool WaitForSingleImageCaptured(int GetImage_time) override;

		//��ȡ�������
		int GetCameraParaInt(CameraParameter para);

		//��ȡͼƬ����
		int GetCameraData(unsigned char* data) override;

		//���ò���ͼ�����
		int SetShootParams();

		//�������һ��ͼƬ
		int SaveImage(ImageType imagetype) override;

		//�ر����
		int CloseDevice() override;
		
		//ֹͣʵʱ�ɼ�
		int StopGrabbing() override;

		//��ʼʵʱ�ɼ�
		int StartGrabbing() override;

			// 设置/获取曝光时间（微秒）
			int SetExposureTime(double exposure);
			double GetExposureTime();

			// 设置/获取增益
			int SetGain(double gain);
			double GetGain();   // ����
	private:
		// ����ץͼ�ĺ�̨�̺߳���
		void GrabLoop();

		// Pylon �������ʹ������ָ�����ͷ�ļ�������
		std::unique_ptr<Pylon::CBaslerUsbInstantCamera> m_camera;
		std::unique_ptr<Pylon::CImageFormatConverter> m_converter;
		std::unique_ptr<Pylon::CPylonImage> m_pylonImage;

		cv::Mat m_frame;              // ����һ֡��BGR��ʽ��
		std::mutex m_frameMutex;      // ���� m_frame
		std::thread m_grabThread;     // ��̨ץͼ�߳�
		std::atomic<bool> m_grabbing; // ����ץͼ�߳�����
		//struct BaslerPimple;
		shared_ptr<BaslerPimple> pBaslerMember;
	};
}
#endif
