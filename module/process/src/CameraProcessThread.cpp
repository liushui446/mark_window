#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <processthreadsapi.h>
#include <iosfwd>
#include <sstream>
#include <exception>
#include <opencv2/opencv.hpp>

#include "process/CameraProcessThread.hpp"
#include "camera/CameraManager.hpp"
#include <Windows.h>
#include <timeapi.h>

#pragma comment(lib, "winmm.lib")

namespace sm {

	AtomicIntData::AtomicIntData(AtomicIntData::value_type&& status)
	{
		AtomicIntData::value_type v = std::forward<AtomicIntData::value_type>(status);
		_Data.store(v, std::memory_order_release);
	}

	AtomicIntData::value_type AtomicIntData::GetValue() throw()
	{
		return _Data.load(std::memory_order_acquire);
	}

	AtomicIntData::value_type AtomicIntData::GetValue() const throw()
	{
		return _Data.load(std::memory_order_acquire);
	}

	void AtomicIntData::SetValue(AtomicIntData::value_type&& value) throw()
	{
		AtomicIntData::value_type v = std::forward<AtomicIntData::value_type>(value);
		_Data.store(v, std::memory_order_release);
	}

	bool AtomicIntData::CAS(AtomicIntData::value_type& expected, AtomicIntData::value_type desired, std::memory_order mem_order)
	{
		return _Data.compare_exchange_weak(expected, desired, mem_order);
	}

	AtomicIntData& AtomicIntData::operator=(AtomicIntData& rhs)
	{
		return *this;
	}

	//后续方便扩展为多线程
	static size_t GetNumThread() {
		return 1;
	}

	struct CameraProcessThread::Pimple {
	public:
		enum class ThreadStatus {
			UNINIT = -1,
			DORMANT = 0,	 // 休眠
			READY = 1,		 // 准备
			BUSY = 2,		 // 忙碌
			INTERRUPTED = 3, // 打断
			QUIT = 4,		 // 退出
		};
	

		int TransThreadStatus2Int(ThreadStatus value)
		{
			switch (value)
			{
			case sm::CameraProcessThread::Pimple::ThreadStatus::UNINIT:
				return -1;
				break;
			case sm::CameraProcessThread::Pimple::ThreadStatus::DORMANT:
				return 0;
				break;
			case sm::CameraProcessThread::Pimple::ThreadStatus::READY:
				return 1;
				break;
			case sm::CameraProcessThread::Pimple::ThreadStatus::BUSY:
				return 2;
				break;
			case sm::CameraProcessThread::Pimple::ThreadStatus::INTERRUPTED:
				return 3;
				break;
			case sm::CameraProcessThread::Pimple::ThreadStatus::QUIT:
				return 4;
				break;
			default:
				break;
			}
		}

		std::atomic<bool> bThreadWait;                      //重构结束标志位
		std::atomic<bool> bThreadError;						//线程中遇到问题（相机拍照失败）返回


		std::thread* pThreads_; // 线程指针
		AtomicIntData ThdStats_;		  //线程的状态
		double dScaleX; // 相机的X方向刻度
		double dScaleY; // 相机的Y方向刻度
		int iImgWidth;  // 图片的宽度
		int iImgHeight; // 图片的高度
		int iImgSize;   // 图片的大小

		cv::Mat bgrMat;

		Pimple()
			: bThreadWait(true)
			, bThreadError(false)

			, pThreads_(nullptr)
			, ThdStats_(static_cast<int>(CameraProcessThread::Pimple::ThreadStatus::READY))
			, dScaleX()
			, dScaleY()
			, iImgWidth(CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_WIDTH))
			, iImgHeight(CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_HEIGHT))
			, iImgSize(CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_SIZE))

			, bgrMat()
		{
			bgrMat = cv::Mat::zeros(1, 1, CV_8UC3);
		}

		~Pimple() {
		}

	};

	CameraProcessThread& CameraProcessThread::Ins() {
		static CameraProcessThread instance;
		return instance;
	}

	CameraProcessThread::CameraProcessThread()
	{
		pMem_ = std::make_shared<Pimple>();
		this->Init();
	}

	CameraProcessThread::~CameraProcessThread() {
		UnInit();
		if ((pMem_->pThreads_) != nullptr) {
			delete (pMem_->pThreads_);
			pMem_->pThreads_ = nullptr;
		}
	}

	void CameraProcessThread::Init() {

		if (pMem_->pThreads_ != nullptr) {
			cout << "CameraProcessThread Init(). Thread has started!" << endl;
			return;
		}

		pMem_->pThreads_ = new std::thread(std::bind(&CameraProcessThread::ThreadFunc, this));

		SetThreadDescription(pMem_->pThreads_->native_handle(), L"CameraProcessThread_");

		cout << "CameraProcessThread Init(). Thread Start!" << endl;
		return;
	}

	void CameraProcessThread::UnInit() {

		if (pMem_->pThreads_ == nullptr) {
			return;
		}

		{
			std::unique_lock<std::mutex> lk(m_mut_thread);
			pMem_->ThdStats_.SetValue(static_cast<int>(CameraProcessThread::Pimple::ThreadStatus::QUIT));
			m_con_var_.notify_one();
		}

		if (pMem_->pThreads_->joinable()) {
			pMem_->pThreads_->join();
		}

		cout << "CameraProcessThread UnInit(). Thread End!" << endl;
		return;
	}

	bool CameraProcessThread::WakeUpAThread()
	{
		int threadState = pMem_->ThdStats_.GetValue();
		if (threadState != static_cast<int>(Pimple::ThreadStatus::DORMANT))
		{
			return false;
		}

		//CameraManager::GetInstance().SetShootParams(CameraID::CAMERA_ID_MAIN);
		
		int try_times = 0;
		do
		{
			try_times++;
			if (try_times > 10)
			{
				std::cerr << "CameraProcessThread::WakeUpAThread() FAIL. Thread State Set Ready FAIL!" << endl;
				return false;
			}
			timeBeginPeriod(1);
			std::this_thread::sleep_for(std::chrono::microseconds(1));
			timeEndPeriod(1);
		} while (
			(!pMem_->ThdStats_.CAS(threadState, static_cast<int>(Pimple::ThreadStatus::READY))) && (threadState == static_cast<int>(Pimple::ThreadStatus::DORMANT)));

		{
			std::unique_lock<std::mutex> lk(m_mut_thread);
			pMem_->bThreadWait = false;
			pMem_->bThreadError = false;
			m_con_var_.notify_one();
		}
		cout << "CameraProcessThread::WakeUpAThread()" << endl;
		return true;
	}

	bool CameraProcessThread::Interrupted()
	{
		if (pMem_->ThdStats_.GetValue() != static_cast<int>(Pimple::ThreadStatus::BUSY))
		{
			cout << "CameraProcessThread::Interrupted() fail. Thread not [Busy]!" << endl;
			return false;
		}

		pMem_->ThdStats_.SetValue(static_cast<int>(Pimple::ThreadStatus::INTERRUPTED));

		return true;
	}

	bool CameraProcessThread::WaitForSingleThreadFinish(unsigned int _numFovs, int _dwMilliseconds)
	{
		try
		{
			int Millis = 0;
			while (true)
			{
				Millis++;
				if (m_finish_flag.load())
				{
					break;
				}

				if (Millis >= _dwMilliseconds)  //等待时间过长报错
				{
					ResetState();
					cout << "CameraProcessThread::WaitForSingleThreadFinish(). Thread Time out!!" << endl;
					return false;
				}

				timeBeginPeriod(1);
				std::this_thread::sleep_for(std::chrono::microseconds(1));
				timeEndPeriod(1);
			}

			// 检查是否有线程错误
			if (pMem_->bThreadError)
			{
				ResetState();
				std::cerr << "CameraProcessThread::WaitForSingleThreadFinish(). Thread Camera Capture Error!" << endl;
				return false;
			}
			ResetState();
			cout << "CameraProcessThread::WaitForSingleThreadFinish() Out" << endl;
		}
		catch (std::exception& e)
		{
			ResetState();
			std::cerr << "CameraProcessThread::WaitForAllThreadHandleFinish, abnormal crash!" << endl;
			return false;
		}
		return true;
	}

	bool CameraProcessThread::StartWork()
	{
		if (pMem_ != nullptr)
		{
			std::unique_lock<std::mutex> lk(m_mut_thread);

			//CameraManager::GetInstance().SetCameraReconMode(CameraID::CAMERA_ID_MAIN, true);
			double scalex;
			double scaley;
			int width = CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_WIDTH);
			int height = CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_HEIGHT);
			int size = CameraManager::GetInstance().GetParaInt(CameraID::CAMERA_ID_MAIN, CameraParameter::CAMERA_PARA_SIZE);

			pMem_->dScaleX = std::move(scalex);
			pMem_->dScaleY = std::move(scaley);
			pMem_->iImgWidth = std::move(width);
			pMem_->iImgHeight = std::move(height);
			pMem_->iImgSize = std::move(size);

			pMem_->bThreadWait.store(true);
			pMem_->bThreadError.store(false);
			m_con_var_.notify_one();

			while (pMem_->ThdStats_.GetValue() != static_cast<int>(Pimple::ThreadStatus::DORMANT))
			{
				timeBeginPeriod(1);
				Sleep(1);
				timeEndPeriod(1);
			}
			//pMem_->num.store(0);
		}
		else
		{
			return false;
		}
		return true;
	}

	bool CameraProcessThread::ResetState()
	{
		if (pMem_ != nullptr)
		{
			m_finish_flag.store(false);
			pMem_->bThreadWait.store(true);
			pMem_->bThreadError.store(false);
			while (pMem_->ThdStats_.GetValue() != static_cast<int>(Pimple::ThreadStatus::DORMANT))
			{
				timeBeginPeriod(1);
				Sleep(1);
				timeEndPeriod(1);
			}
			//pMem_->num.store(0);
			//CameraManager::GetInstance().ClearReconBuffer(CameraID::CAMERA_ID_MAIN);
		}
		else
		{
			return false;
		}
		return true;
	}

	void CameraProcessThread::GetBGRMat(cv::Mat& bgrMat)
	{
		bgrMat = pMem_->bgrMat.clone();
	}

	void CameraProcessThread::ThreadFunc()
	{
		std::vector<cvMatPtr> temp_mat_ptrs;
		uchar* temp_data = nullptr;
		cvMatPtr temp_ptr = nullptr;
		ProduceMatsPtr temp_produce_mats = std::make_shared <ProduceMats>();
		std::string str = "";
		auto start = std::chrono::high_resolution_clock::now();
		bool isLightOn = false;
		while (true)
		{
			{
				std::unique_lock<std::mutex> lk(m_mut_thread);
				// 子进程的中wait函数对互斥量进行解锁，同时线程进入阻塞或者等待状态。
				// 当唤醒当前线程锁时，检查缓冲区中是否有待检测数据，如果有，或者线程要关闭，线程开启，否则线程继续等待
				// 线程开启与否，由重构拍照结束标志位决定，当重构拍照结束标志位false时，会调用唤醒函数。
				if (pMem_->bThreadWait)
				{
					temp_mat_ptrs.clear();
					temp_data = nullptr;
					temp_produce_mats->vProduceMats.clear();
					temp_ptr.reset();
				}

				m_con_var_.wait(lk, [&]()
					{
						if (static_cast<int>(Pimple::ThreadStatus::QUIT) != pMem_->ThdStats_.GetValue())
						{
							pMem_->ThdStats_.SetValue(static_cast<int>(Pimple::ThreadStatus::DORMANT));
						}
						return !(pMem_->bThreadWait) || (static_cast<int>(Pimple::ThreadStatus::QUIT) == pMem_->ThdStats_.GetValue());
					});
			}

			// 判断是否是退出线程的命令
			if (static_cast<int>(Pimple::ThreadStatus::QUIT) == pMem_->ThdStats_.GetValue())
			{
				cout << "CameraProcessThread::ThreadFunc() quit." << endl;
				pMem_->bThreadError.store(false);
				pMem_->bThreadWait.store(true);
				m_finish_flag.store(true);
				break;
			}

			// 判断是否是中断线程的命令
			if (static_cast<int>(Pimple::ThreadStatus::INTERRUPTED) == pMem_->ThdStats_.GetValue())
			{
				cout << "CameraProcessThread::ThreadFunc() Interrupted." << endl;
				temp_mat_ptrs.clear();
				temp_produce_mats->vProduceMats.clear();
				pMem_->bThreadError.store(false);
				pMem_->bThreadWait.store(true);
				m_finish_flag.store(true);
				m_con_finish_.notify_all();
				continue;
			}

			// 计时开始
			start = std::chrono::high_resolution_clock::now();

			try
			{
				//工作区
				pMem_->ThdStats_.SetValue(static_cast<int>(Pimple::ThreadStatus::BUSY));

				Sleep(15);
				CameraManager::GetInstance().SoftWareTrigger(CameraID::CAMERA_ID_MAIN);
				if (!CameraManager::GetInstance().WaitForGrabOneImageFinish(CameraID::CAMERA_ID_MAIN, 1000))
				{
					continue;
				}

				// 获得图像首地址
				if (temp_data == nullptr)
				{
					timeBeginPeriod(1);
					Sleep(1);
					timeEndPeriod(1);
					continue;
				}

				//创建容器
				temp_ptr.reset(new cv::Mat(pMem_->iImgHeight, pMem_->iImgWidth, CV_8UC3));
				//temp_ptr.reset(new cv::Mat(pMem_->iImgHeight, pMem_->iImgWidth, CV_8UC1));
				memcpy_s(temp_ptr->data, pMem_->iImgSize, temp_data, pMem_->iImgSize);

				// 是否需要转换成灰度图像
				cv::cvtColor(*temp_ptr, *temp_ptr, cv::COLOR_BGR2GRAY);//彩图格式是否是RGB顺序需要查看配置
				cv::cvtColor(*temp_ptr, *temp_ptr, cv::COLOR_BayerRG2GRAY);

				//计数并回馈
				temp_mat_ptrs.push_back(std::move(temp_ptr));
				temp_ptr.reset();


				//ImgCirBuf::getInstance()->push_back(temp_produce_mats);
				//temp_produce_mats = nullptr;
				temp_produce_mats = std::make_shared <ProduceMats>();
				m_finish_flag.store(true);
	

				auto stop = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); // 毫秒
			}
			catch (const std::exception& e)
			{
				std::cerr << "CameraProcessThread::WaitForAllThreadHandleFinish, abnormal crash!" << endl;
			}
		}

		temp_mat_ptrs.clear();
		temp_data = nullptr;
		temp_ptr.reset();
		temp_produce_mats.reset();
		str = "";
		return;
	}

};