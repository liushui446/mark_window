#pragma once

#ifdef NCC_MATCH_DLL_EXPORTS
#define NCC_MATCH_API __declspec(dllexport)
#else
#define NCC_MATCH_API __declspec(dllimport)
#endif

#include <cstddef>

// 匹配结果结构体
struct NccMatchResult {
    double x;           // 匹配中心X坐标
    double y;           // 匹配中心Y坐标
    double angle;       // 匹配角度（度）
    float similarity;   // 匹配相似度（0-1）
    float time_ms;
    bool success;       // 是否匹配成功
};
struct NccRect {
    int x;          // 左上角X坐标
    int y;          // 左上角Y坐标
    int width;      // 宽度
    int height;     // 高度
};
// C风格导出接口
extern "C" {
    /**
     * @brief 生成模板并保存边缘点到XML文件
     * @param templateImagePath 模板图像路径
     * @param edgeXmlPath 边缘点XML文件保存路径
     * @param angleStep 角度步长（度）
     * @param minAngle 最小角度（度）
     * @param maxAngle 最大角度（度）
     * @return 是否成功
     */
    NCC_MATCH_API bool NCC_CreateTemplate(
        const char* templateImagePath,
        const char* edgeXmlPath,
        const NccRect* roi,
        int angleStep = 5,
        int minAngle = -60,
        int maxAngle = 60
        
    );

    /**
     * @brief 使用保存的边缘点XML文件执行匹配
     * @param testImagePath 待检测图像路径
     * @param edgeXmlPath 边缘点XML文件路径
     * @param result 匹配结果输出
     * @return 是否成功
     */
    NCC_MATCH_API bool NCC_PerformMatching(
        const char* testImagePath,
        const char* edgeXmlPath,
        const NccRect* roi,
        NccMatchResult* result
    );

    /**
     * @brief 设置Canny边缘检测参数
     * @param thresh1 低阈值
     * @param thresh2 高阈值
     * @param apertureSize 孔径大小
     * @param L2gradient 是否使用L2梯度
     */
    //NCC_MATCH_API void NCC_SetCannyParams(
    //    double thresh1 = 170,
    //    double thresh2 = 200,
    //    int apertureSize = 3,
    //    bool L2gradient = true
    //);

    ///**
    // * @brief 设置轮廓面积阈值
    // * @param threshold 面积阈值
    // */
    //NCC_MATCH_API void NCC_SetContourAreaThreshold(double threshold = 8.0);

}
