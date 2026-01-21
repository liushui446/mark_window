#include <stdio.h>
#include <Windows.h>
#include <conio.h>
#include <iostream>
#include <string>
#include <iostream>
#include <filesystem>

// Include files to use OpenCV API.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#include <pylon/usb/BaslerUsbInstantCamera.h>

#include "camera/camera.hpp"
//#include "CameraManager.hpp"
#include "log/log.hpp"
#include "core/core.hpp"


using namespace Pylon;
using namespace GenApi;
using namespace cv;
using namespace std;

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 1;



namespace
{
    bool isSystemOnline()
    {
        
    }
}

namespace sm
{
    namespace camera
    {
        std::shared_ptr<camera::CameraManager> CameraManager::pIns(new camera::CameraManager);

        CameraManager& CameraManager::GetInstance() {
            return *pIns;
        }

        CameraManager::~CameraManager()
        {
        }

        CameraManager::CameraManager() {}

        void CameraManager::UseBaslerCam()
        {
            // Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
            // is initialized during the lifetime of this object.
            Pylon::PylonAutoInitTerm autoInitTerm;

            try
            {
                // Create an instant camera object with the camera device found first.
                // 1. 连接并初始化 Basler USB 相机，配置曝光时间、增益、伽马等参数；
                cout << "Creating Camera..." << endl;
                CBaslerUsbInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());

                // Print the model name of the camera.
                cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

                // Open the camera.
                camera.Open();

                // Set exposure time to ensure 30 FPS.
                INodeMap& nodeMap = camera.GetNodeMap();
                CFloatPtr exposureTime = nodeMap.GetNode("ExposureTime");
                if (IsWritable(exposureTime))
                {
                    exposureTime->SetValue(40000.0); // 将曝光时间设置为10000微秒
                }
                else
                {
                    cerr << "ExposureTime is not writable." << endl;
                }

                // Set gain.
                CFloatPtr gain = nodeMap.GetNode("Gain");
                if (IsWritable(gain))
                {
                    gain->SetValue(10.0); // 将增益设置为10 dB
                }
                else
                {
                    cerr << "Gain is not writable." << endl;
                }

                // Set gamma.
                CFloatPtr gamma = nodeMap.GetNode("Gamma");
                if (IsWritable(gamma))
                {
                    gamma->SetValue(1.2); // 将伽马值设置为1.2
                }
                else
                {
                    cerr << "Gamma is not writable." << endl;
                }

                //2. 实时采集相机图像，将 Basler 相机的原始图像格式转换为 OpenCV 可处理的 BGR 格式；
                // Create pylon image format converter and pylon image.
                CImageFormatConverter formatConverter;
                formatConverter.OutputPixelFormat = PixelType_BGR8packed;
                CPylonImage pylonImage;

                // Create an OpenCV image.
                Mat openCvImage;

                // Start grabbing images.
                camera.StartGrabbing(GrabStrategy_LatestImageOnly);

                // This smart pointer will receive grab result data.
                CGrabResultPtr ptrGrabResult;

                //3. 将采集的图像序列保存为 AVI 格式的视频文件；
                // Create video writer.
                VideoWriter videoWriter;
                int codec = VideoWriter::fourcc('M', 'J', 'P', 'G'); // Select video codec
                double fps = 30.0; // Set frame rate
                string videoFileName = "D:/video.avi"; // Video file name
                bool isVideoWriterOpen = false;

                // Frame counter for saving every 5th frame
                int frameCounter = 0;

                // Variables for frame rate calculation
                int frameCount = 0;
                double totalTime = 0.0;
                double startTime = static_cast<double>(getTickCount());

                // Camera.StopGrabbing() is called by RetrieveResult() method when c_countOfImagesToGrab images have been retrieved.
                while (camera.IsGrabbing())
                {
                    // 等待采集结果，超时 5000 毫秒，超时则抛异常
                    // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                    camera.RetrieveResult(5000, ptrGrabResult, TimeoutHandling_ThrowException);

                    // Image grabbed successfully?
                    if (ptrGrabResult->GrabSucceeded())// 采集成功
                    {
                        // Access the image data.
                        cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
                        cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
                        const uint8_t* pImageBuffer = (uint8_t*)ptrGrabResult->GetBuffer();
                        cout << "Gray value of first pixel: " << (uint32_t)pImageBuffer[0] << endl << endl;

                        // Convert the grabbed buffer to pylon image.
                        formatConverter.Convert(pylonImage, ptrGrabResult);
                        // Create an OpenCV image out of pylon image.
                        openCvImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());

                        // 3. 打开视频写入器（首次采集成功时）
                        if (!isVideoWriterOpen)
                        {
                            videoWriter.open(videoFileName, codec, fps, openCvImage.size(), true);
                            if (!videoWriter.isOpened())
                            {
                                cerr << "Could not open the output video file for write" << endl;
                                return;
                            }
                            isVideoWriterOpen = true;
                        }

                        // 4. 写入当前帧到视频文件
                        videoWriter.write(openCvImage);

                        //5. 每采集 5 帧图像就保存一张 PNG 格式的截图；
                        // Save every 5th frame as an image
                        if (frameCounter % 5 == 0)
                        {
                            string imageFileName = "D:/picture/frame_" + to_string(frameCounter) + ".png";
                            imwrite(imageFileName, openCvImage);
                            cout << "Saved frame " << frameCounter << " as " << imageFileName << endl;
                        }

                        // 6. 计算并输出实时帧率
                        // Increment frame counter
                        frameCounter++;
                        // Calculate frame rate
                        frameCount++;
                        double currentTime = static_cast<double>(getTickCount());
                        double elapsedTime = (currentTime - startTime) / getTickFrequency();
                        totalTime += elapsedTime;
                        double frameRate = frameCount / totalTime;
                        cout << "Frame rate: " << frameRate << " fps" << endl;
                        // Reset start time for next frame
                        startTime = currentTime;

                        //7.实时显示采集到的图像画面；
                        // Create a display window.
                        namedWindow("OpenCV Display Window", WINDOW_NORMAL);
                        // Display the current image with OpenCV.
                        imshow("OpenCV Display Window", openCvImage);
                        // Define a timeout for customer's input in ms.
                        // '0' means indefinite, i.e. the next image will be displayed after closing the window 
                        // '1' means live stream.
                        if (waitKey(1) >= 0)
                        {
                            break;
                        }
                    }
                    else
                    {
                        cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
                    }
                }

                // Release the video writer.
                if (isVideoWriterOpen)
                {
                    videoWriter.release();
                }

                // Close the camera.
                camera.Close();
            }
            catch (GenICam::GenericException& e)
            {
                // Error handling.
                cerr << "An exception occurred." << endl
                    << e.GetDescription() << endl;
            }

            // Wait for user input to exit.
            cerr << "Press Enter to exit." << endl;
            while (cin.get() != '\n');

            // Return
            // Wait for user input to exit.
            cerr << "Press Enter to exit." << endl;
            while (cin.get() != '\n');

            return;
        }

        void CameraTest()
        {

        }     
    }
} // namespace

