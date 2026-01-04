#define MARK_EXPORTS
#include "MarkWrapper.h"
#include <algorithm/markInterface.h>
#include <cstring>
#include <algorithm>  
#include <opencv2/opencv.hpp>  // 用于图像操作

// 辅助函数：获取图像宽高
static bool get_image_dimensions(const char* image_path, int* width, int* height) {
    if (!image_path || !width || !height) {
        return false;
    }

    // 使用OpenCV读取图像获取尺寸
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        return false;
    }

    *width = image.cols;
    *height = image.rows;
    return true;
}

// 辅助函数：验证ROI有效性（确保不超出图像范围，内部使用）
static void validateRoi(const MarkRect* input_roi, int img_width, int img_height, MarkRect& valid_roi) {
    if (!input_roi) {
        // ROI为空时使用全图
        valid_roi = { 0, 0, img_width, img_height };
        return;
    }

    // 边界修正：确保ROI在图像范围内
    valid_roi.x = std::max(0, input_roi->x);
    valid_roi.y = std::max(0, input_roi->y);
    valid_roi.width = std::min(input_roi->width, img_width - valid_roi.x);
    valid_roi.height = std::min(input_roi->height, img_height - valid_roi.y);

    // 确保ROI尺寸有效（最小10x10）
    valid_roi.width = std::max(valid_roi.width, 10);
    valid_roi.height = std::max(valid_roi.height, 10);
}

// 析构函数：释放资源
MARK_WRAPPER::~MARK_WRAPPER() {
    is_initialized_ = false;
    matcher_ = nullptr;
    memset(last_error_, 0, sizeof(last_error_));
}

// 1. 生成模板（支持ROI，可空）
bool MARK_WRAPPER::GenerateTemplate(const MarkTemplateConfig& config) {
    // 参数合法性检查
    if (!config.image_path || !config.save_dir || !config.model_name) {
        strcpy_s(last_error_, "模板生成失败：输入路径或名称为空");
        return false;
    }

    // 获取图像宽高
    int img_width = 0, img_height = 0;
    if (!get_image_dimensions(config.image_path, &img_width, &img_height)) {
        strcpy_s(last_error_, "模板生成失败：无法获取图像尺寸");
        return false;
    }

    // 验证ROI有效性
    MarkRect valid_roi;
    validateRoi(config.roi, img_width, img_height, valid_roi);

    // 调用支持ROI的模板生成接口
    bool result = GenerateMarkTemplate(
        config.image_path,
        config.save_dir,
        config.model_name,
        config.binarize_thresh,
        config.roi ? &valid_roi : nullptr  // 空ROI传递nullptr
    );

    // 设置错误信息
    if (!result) {
        strcpy_s(last_error_, config.roi ?
            "模板生成失败：ROI无效或图像处理错误" :
            "模板生成失败：图像加载或模板创建错误");
    }
    else {
        strcpy_s(last_error_, config.roi ?
            "模板生成成功（使用指定ROI）" :
            "模板生成成功（使用全图）");
    }
    return result;
}

// 2. 初始化匹配器
bool MARK_WRAPPER::InitMatcher(const char* model_dir, const char* model_name) {
    if (!model_dir || !model_name) {
        strcpy_s(last_error_, "初始化失败：模型路径或名称为空");
        return false;
    }

    bool result = InitMarkMatcher(model_dir, model_name);
    if (result) {
        is_initialized_ = true;
        strcpy_s(last_error_, "匹配器初始化成功");
    }
    else {
        strcpy_s(last_error_, "初始化失败：模型加载错误");
    }
    return result;
}

// 3. 执行单图匹配（支持ROI，可空）
bool MARK_WRAPPER::RunMatching(const MarkMatchConfig& config, MarkMatchResult& result) {
    // 初始化检查
    if (!is_initialized_) {
        strcpy_s(last_error_, "匹配失败：请先初始化匹配器");
        return false;
    }

    // 参数检查
    if (!config.image_path || !config.output_image) {
        strcpy_s(last_error_, "匹配失败：图像路径为空");
        return false;
    }

    // 初始化结果
    memset(&result, 0, sizeof(MarkMatchResult));
    result.is_success = false;

    // 获取图像宽高
    int img_width = 0, img_height = 0;
    if (!get_image_dimensions(config.image_path, &img_width, &img_height)) {
        strcpy_s(last_error_, "匹配失败：无法获取图像尺寸");
        return false;
    }

    // 验证ROI有效性
    MarkRect valid_roi;
    validateRoi(config.roi, img_width, img_height, valid_roi);

    // 调用支持ROI的匹配接口
    double x, y, r, time_ms;
    float similarity;
    int binarize_thresh;

    bool match_result = RunMarkMatchingSingle(
        config.image_path,
        config.output_image,
        &x, &y, &r,
        &similarity,
        &time_ms,
        &binarize_thresh,
        config.roi ? &valid_roi : nullptr  // 空ROI传递nullptr
    );

    // 填充结果
    if (match_result) {
        result.x = x;
        result.y = y;
        result.angle = r;
        result.similarity = similarity;
        result.time_ms = time_ms;
        result.binarize_thresh = binarize_thresh;
        result.is_success = true;
        strcpy_s(last_error_, config.roi ?
            "匹配成功（使用指定ROI）" :
            "匹配成功（使用全图）");
    }
    else {
        strcpy_s(last_error_, config.roi ?
            "匹配失败：ROI内未找到有效结果" :
            "匹配失败：未找到有效结果");
    }
    return match_result;
}
