#include "camera/CameraManager.hpp"
#include "camera/BaslerCamera.hpp"

#include <pylon/PylonIncludes.h>
#include <pylon/usb/PylonUsbIncludes.h>
#include <pylon/EnumParameterT.h>
#include <opencv2/opencv.hpp>
using namespace std;

// 全局使用 Pylon 命名空间
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
		//pMembers->AccessByCameraID.insert(pair<CameraID, CameraPtr>(id, cameraPtr));
		pMembers->AccessByCameraID.emplace(id, cameraPtr);
	}

	CameraManager::~CameraManager() {
		
	}

	int CameraManager::Init(CameraID id) {

		if (IDtoCamPtr(id) != nullptr) {
			return IDtoCamPtr(id)->Init();
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
        // 1. 初始化 Pylon 运行时（显式指定 USB 传输层）
        PylonInitialize();

        try {
            // 2. 获取传输层工厂，优先加载 USB 传输层
            CTlFactory& tlFactory = CTlFactory::GetInstance();

            // ========== 关键：只枚举 USB 传输层的设备 ==========
            ITransportLayer* pUsbTl = tlFactory.CreateTl("BaslerUsb"); // USB 传输层标识
            if (pUsbTl == nullptr) {
                cerr << "无法加载 USB 传输层！请检查 pylon SDK 安装是否包含 USB 组件" << endl;
                PylonTerminate();
                return -1;
            }

            // 3. 枚举 USB 传输层下的所有相机（确保只找 USB 相机）
            DeviceInfoList_t usbDevices;
            pUsbTl->EnumerateDevices(usbDevices);

            if (usbDevices.empty()) {
                cerr << "USB 传输层下未找到任何相机！" << endl;
                tlFactory.ReleaseTl(pUsbTl); // 释放传输层
                PylonTerminate();
                return -1;
            }

            // 4. 打印 USB 相机信息（验证识别结果）
            cout << "找到 " << usbDevices.size() << " 个 USB 相机：" << endl;
            for (size_t i = 0; i < usbDevices.size(); ++i) {
                cout << "相机 " << i + 1 << "：" << endl;
                cout << "  型号：" << usbDevices[i].GetModelName() << endl;
                cout << "  序列号：" << usbDevices[i].GetSerialNumber() << endl;
            }

            // 5. 连接第一个 USB 相机（核心修正：用 USB 传输层的设备创建实例）
            CInstantCamera camera(tlFactory.CreateDevice(usbDevices[0]));
            cout << "成功连接 USB 相机：" << camera.GetDeviceInfo().GetModelName() << endl;

            // 6. 打开相机并配置参数（USB 相机需先 Open 再操作参数）
            camera.Open();

            // ========== 示例：读取宽高 + 设置触发模式（USB 相机适配） ==========
            INodeMap& nodeMap = camera.GetNodeMap();
            // 读取宽度
            CIntegerParameter widthParam(nodeMap, "Width");
            if (widthParam.IsReadable()) {
                cout << "图像宽度：" << widthParam.GetValue() << endl;
            }
            // 读取高度
            CIntegerParameter heightParam(nodeMap, "Height");
            if (heightParam.IsReadable()) {
                cout << "图像高度：" << heightParam.GetValue() << endl;
            }

            // 7. 关闭相机 + 释放传输层
            camera.Close();
            tlFactory.ReleaseTl(pUsbTl); // 必须释放 USB 传输层
        }
        catch (const GenericException& e) {
            cerr << "USB 相机操作异常：" << e.GetDescription() << endl;
            PylonTerminate();
            return -1;
        }

        // 8. 释放 Pylon 运行时
        PylonTerminate();
        cout << "程序正常退出" << endl;
        return 0;
	}
}