#pragma once
#include <cstddef>
#include <algorithm/markInterface.h>  // 引入统一的MarkRect定义
//// 新增：ROI区域结构体（支持空区域表示全图）
//struct MarkRect {
//    int x;          // 左上角X坐标
//    int y;          // 左上角Y坐标
//    int width;      // 区域宽度
//    int height;     // 区域高度
//};

// 匹配结果结构体
struct MarkMatchResult {
    double x;               // 匹配X坐标
    double y;               // 匹配Y坐标
    double angle;           // 匹配角度
    float similarity;       // 相似度(0~1)
    double time_ms;         // 耗时(毫秒)
    int binarize_thresh;    // 二值化阈值
    bool is_success;        // 是否成功
};

// 模板生成配置（新增ROI参数）
struct MarkTemplateConfig {
    const char* image_path;      // 输入图像路径
    const char* save_dir;        // 模板保存目录
    const char* model_name;      // 模板名称
    int binarize_thresh = 160;   // 二值化阈值(默认160)
    const MarkRect* roi = nullptr; // ROI区域（默认空表示全图）
};

// 匹配配置（新增ROI参数）
struct MarkMatchConfig {
    const char* image_path;      // 待匹配图像路径
    const char* output_image;    // 输出结果图像路径
    const MarkRect* roi = nullptr; // ROI区域（默认空表示全图）
};

class MARK_WRAPPER {
public:
    MARK_WRAPPER() = default;
    ~MARK_WRAPPER();

    // 生成模板（支持ROI，可空）
    bool GenerateTemplate(const MarkTemplateConfig& config);

    // 初始化匹配器
    bool InitMatcher(const char* model_dir, const char* model_name);

    // 执行单图匹配（支持ROI，可空）
    bool RunMatching(const MarkMatchConfig& config, MarkMatchResult& result);

    // 获取最后一次错误信息
    const char* GetLastError() const { return last_error_; }

private:
    MARK_WRAPPER(const MARK_WRAPPER&) = delete;
    MARK_WRAPPER& operator=(const MARK_WRAPPER&) = delete;

    void* matcher_ = nullptr;    // 隐藏底层匹配器
    bool is_initialized_ = false; // 初始化状态
    char last_error_[512] = { 0 };  // 错误信息缓存
};