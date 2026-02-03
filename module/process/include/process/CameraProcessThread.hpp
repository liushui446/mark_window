#pragma once
#include <ppl.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include"core/CommonCore.hpp"

namespace sm {

	typedef std::shared_ptr<cv::Mat> cvMatPtr;

	class AtomicIntData
	{
	public:
		typedef int value_type;

	private:
		std::atomic<int> _Data;
		AtomicIntData& operator = (AtomicIntData& rhs);

	public:
		explicit AtomicIntData(AtomicIntData::value_type&& size = 1);
		value_type GetValue() const throw();
		value_type GetValue() throw();
		void SetValue(value_type&& value) throw();

		bool CAS(value_type& expected, value_type desired, std::memory_order mem_order = std::memory_order::memory_order_seq_cst);
	};

	struct ProduceMats
	{
	public:
		std::vector<cvMatPtr> vProduceMats;
		int num;
		unsigned int id;

		ProduceMats()
		{
			vProduceMats.clear();
			num = 0;
			id = 0;
		}
	};
	using ProduceMatsPtr = std::shared_ptr<ProduceMats>;//modify

	class SM_EXPORTS CameraProcessThread
	{
	public:
		static CameraProcessThread& Ins();

	private:
		CameraProcessThread();
		~CameraProcessThread();

	public:
		void Init();
		void UnInit();

		bool WakeUpAThread();
		bool Interrupted();
		bool WaitForSingleThreadFinish(unsigned int _numFovs, int _dwMilliseconds);

		//bool SetLightValue(LightValue value);
		//LightValue GetLightValue();
		bool StartWork();
		bool ResetState();	 // 复位相机和采图线程的状态
		void GetBGRMat(cv::Mat& bgrMat);

	private:
		void ThreadFunc();

	private:
		struct Pimple;
		std::shared_ptr<Pimple> pMem_;
		std::condition_variable m_con_var_;
		std::condition_variable m_con_finish_;
		std::mutex m_mut_thread;
		std::mutex m_mut_cpatured;
		std::atomic<bool> m_finish_flag{ false };
	public:
		CameraProcessThread(const CameraProcessThread&) = delete;
		CameraProcessThread(CameraProcessThread&&) = delete;
		CameraProcessThread& operator=(const CameraProcessThread&) = delete;
		CameraProcessThread& operator=(CameraProcessThread&&) = delete;
	};

	
}