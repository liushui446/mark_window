#pragma once

namespace sm
{
    namespace camera {
		class CameraManager
		{
		public:
			static CameraManager& GetInstance();//µ¥Àý
			~CameraManager();

			void UseBaslerCam();

		private:
			CameraManager();

			static std::shared_ptr<camera::CameraManager> pIns;

		};

        void CameraTest();
    }
}
