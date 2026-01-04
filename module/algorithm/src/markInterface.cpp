#define MARK_EXPORTS

#include "KcgMatch.h"
#include "CannyEdge.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm/markInterface.h>
#include <chrono>
#include <memory>
#include <cerrno>

using namespace kcg;
#ifdef _WIN32
#include <direct.h>  // _mkdir
#else
#include <sys/stat.h> // mkdir
#endif

// 辅助函数：将MarkRect转换为cv::Rect
static cv::Rect markRectToCvRect(const MarkRect* markRoi) {
    if (!markRoi) {
        return cv::Rect(); // 返回空矩形表示全图
    }
    return cv::Rect(markRoi->x, markRoi->y, markRoi->width, markRoi->height);
}

bool create_directory_if_not_exists(const std::string& path) {
#ifdef _WIN32
    int ret = _mkdir(path.c_str());
    return ret == 0 || errno == EEXIST;
#else
    int ret = mkdir(path.c_str(), 0777);
    return ret == 0 || errno == EEXIST;
#endif
}

static std::unique_ptr<KcgMatch> g_matcher;
static bool g_model_loaded = false;

extern "C" {

    /// ① 生成模板
    MARK_API bool GenerateMarkTemplate(
        const char* image_path,
        const char* save_model_dir,
        const char* model_name,
        int binarize_thresh,     // 二值化阈值
        const MarkRect* roi      // 自定义ROI结构体，传NULL表示全图
    ) {
        if (!image_path || !save_model_dir || !model_name) return false;

        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return false;
        }

        cv::Mat gray;
        if (image.channels() == 3)
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = image.clone();

        if (!create_directory_if_not_exists(save_model_dir)) {
            std::cerr << "Failed to create directory: " << save_model_dir << std::endl;
            return false;
        }

        // 转换ROI并验证
        cv::Rect cv_roi = markRectToCvRect(roi);
        cv::Mat roi_gray;

        // 检查ROI有效性（与图像边界交集）
        if (cv_roi.area() > 0) {
            // 计算与图像的有效交集
            cv::Rect valid_roi = cv_roi & cv::Rect(0, 0, gray.cols, gray.rows);
            if (valid_roi.area() > 0) {
                roi_gray = gray(valid_roi);
            }
            else {
                std::cerr << "ROI is out of image bounds, using full image" << std::endl;
                roi_gray = gray;
            }
        }
        else {
            // 无有效ROI，使用全图
            roi_gray = gray;
        }

        KcgMatch matcher(save_model_dir, model_name);
        AngleRange ar(-6.f, 6.f, 1.f);
        ScaleRange sr(1.0f, 1.0f, 0.0f);
        matcher.MakingTemplates(roi_gray, ar, sr, 0, binarize_thresh, 30.f, 60.f);

        return true;
    }

    MARK_API bool InitMarkMatcher(const char* model_dir, const char* model_name) {
        if (!model_dir || !model_name) return false;

        g_matcher = std::make_unique<KcgMatch>(model_dir, model_name);
        g_matcher->LoadModel();
        g_model_loaded = true;
        return g_model_loaded;
    }

    MARK_API bool RunMarkMatchingSingle(
        const char* search_image_path,
        const char* output_path,
        double* x,
        double* y,
        double* r,
        float* similarity,
        double* time_ms,
        int* binarize_thresh,
        const MarkRect* roi                // 传入ROI区域
    ) {
        if (!g_model_loaded || !search_image_path || !output_path ||
            !x || !y || !r || !similarity || !time_ms || !binarize_thresh)
            return false;

        cv::Mat search = cv::imread(search_image_path);
        if (search.empty()) {
            std::cerr << "Failed to load search image: " << search_image_path << std::endl;
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat gray;
        cv::cvtColor(search, gray, cv::COLOR_BGR2GRAY);

        // 转换ROI并验证
        cv::Rect cv_roi = markRectToCvRect(roi);
        cv::Mat roi_gray;

        // 计算有效ROI（与图像边界交集）
        cv::Rect valid_roi = cv_roi & cv::Rect(0, 0, gray.cols, gray.rows);
        if (valid_roi.area() > 0) {
            roi_gray = gray(valid_roi);
        }
        else {
            std::cerr << "ROI is out of image bounds, using full image" << std::endl;
            roi_gray = gray;
        }
        cv::Mat binary_img, edges;
        cv::threshold(roi_gray, binary_img, 180, 255, cv::THRESH_BINARY);
        cv::Canny(binary_img, edges, 140, 200, 3, true);
        int edgeCount= cv::countNonZero(edges);
        // 定义边缘点数量阈值（需根据实际场景调试确定）
        const int THRESH_LOW = 250;    // 边缘点少的阈值
        const int THRESH_MEDIUM = 500;// 边缘点中的阈值
        PyramidLevel level;
        if (edgeCount < THRESH_LOW) {
            level = PyramidLevel_0;    // 边缘点少，用0级金字塔（原图精度优先）
        }
        else if (edgeCount < THRESH_MEDIUM) {
            level = PyramidLevel_1;    // 边缘点中，用1级金字塔（平衡精度与速度）
        }
        else {
            level = PyramidLevel_2;    // 边缘点多，用2级金字塔（速度优先）
        }
        auto matches = g_matcher->Matching(roi_gray, 0.60f, 0.1f, 30.f, 0.9f,
            level, 2, 5);
        if (matches.empty()) return false;

        auto& match = matches[0];
        // 校正坐标（加上ROI偏移）
        match.x += valid_roi.x;
        match.y += valid_roi.y;
        *x = match.x;
        *y = match.y;
        *r = match.r;
        *similarity = match.similarity;

        auto end = std::chrono::high_resolution_clock::now();
        *time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        *binarize_thresh = match.template_id;

        cv::Mat colorImage;
        cv::cvtColor(gray, colorImage, cv::COLOR_GRAY2BGR);
        g_matcher->DrawMatches1(colorImage, matches, cv::Scalar(0, 0, 255));
        //改成以图像为中心
        auto& match1 = matches[0];
        *x = match1.x;
        *y = match1.y;
        *r = match1.r;
        return cv::imwrite(output_path, gray);
    }

} // extern "C"
