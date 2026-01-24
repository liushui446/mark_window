#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class NccMatch {
private:
    // 角度参数
    int angleStep_;       // 角度步长（度）
    int minAngle_;        // 最小角度（度）
    int maxAngle_;        // 最大角度（度）

    // Canny边缘检测参数
    double cannyThresh1_;  // 低阈值
    double cannyThresh2_;  // 高阈值
    int cannyApertureSize_; //  aperture尺寸
    bool cannyL2gradient_;  // 是否使用L2梯度

    // 轮廓过滤参数
    double contourAreaThresh_;  // 最小轮廓面积

    cv::Size maxTemplateSize_;
    cv::Mat downsampledRefImage_;

    std::vector<std::pair<std::vector<cv::Mat>, float>>template_pyrdown_;//原始尺寸模板,压缩2层
    std::vector<std::pair<std::vector<cv::Point2f>, float>> rotatedSubedgePoints_;    // 格式：<旋转后的亚像素点集合, 对应的角度（弧度）>

    // 辅助方法（原有+新增）
    void dftTemp(const cv::Mat& refDownsampled, const cv::Mat& templ, cv::Mat& _dftTempl, int ctype = CV_32F);
  //  bool saveEdgePointsToXml(const std::vector<cv::Point>& edgePoints, const std::string& xmlPath);
    void extractEdgePointsWithNoiseFilter(const cv::Mat& edgeImage, std::vector<cv::Point>& edgePoints, const int& minArea);

    // 模板存储
    std::vector<std::pair<cv::Mat, float>> spatialTemplates_;  // <空间域模板, 角度(弧度)>
    std::vector<cv::Mat> fourierTemplates_;                    // 傅里叶域模板
    cv::Rect baseBbox_;                                        // 基准模板边界

    // 匹配相关参数
    cv::Size dftSize_;    // DFT最优尺寸
    cv::Size corrSize_;   // 互相关结果尺寸
    int maxDepth_;        // 最大数据深度

    // 内部方法
    bool extractEdgePoints(const cv::Mat& grayImage, std::vector<cv::Point>& edgePoints);
    cv::Mat computeFourierTemplate(const cv::Mat& spatialTemplate);
    void dftimg(const cv::Mat& img, cv::Mat& corr, cv::Mat& dftImg);
    void dftimg(const cv::Mat& img, cv::Mat& corr, cv::Mat& _dftImg, cv::Size& corrsize, cv::Size& dftsize, int ctype, int& maxDepth,
        double delta, int borderType);
    //void crossCorr1(const cv::Mat& img, const cv::Mat& tempDft, cv::Mat& result, const cv::Mat& imgDft);
    void crossCorr1(const cv::Mat& img, const cv::Mat& _dftTempl, cv::Mat& corr, const cv::Mat& _dftImg,
        cv::Size dftsize, int ctype, int maxDepth, double delta, int borderType);
    cv::Mat lastEdgeImage_;  // 保存最后一次边缘图像
    bool loadEdgePointsAndGenerateTemplates(const std::string& edgeXmlPath, const cv::Mat& Image);
    bool bestTemplate(const std::vector<cv::Mat>& image, const int& method, int& result);
    std::vector<cv::Point> sparseEdgePointsSimple(const std::vector<cv::Point>& srcEdgePoints,float minDistThreshold);
public:
    // 构造函数
    NccMatch(int angleStep = 1, int minAngle = -5, int maxAngle = 5,
        double cannyThresh1 = 120, double cannyThresh2 = 200,
        double contourAreaThresh = 8.0);

    // 参数设置接口
    void setAngleParams(int angleStep, int minAngle, int maxAngle);
    void setCannyParams(double thresh1, double thresh2, int apertureSize = 3, bool L2gradient = true);
    void setContourAreaThreshold(double threshold);

    // 模板生成接口（不使用ROI）
    bool generateTemplates(const std::vector<cv::Point>& edgePoints, std::vector<cv::Point2f>& subedgePoints,const cv::Mat& refImage);
    //优化时间
    bool generateTemplates1(const std::vector<cv::Point>& edgePoints, std::vector<cv::Point2f>& subedgePoints, const cv::Mat& refImage);
    bool generateTemplateAndSaveEdgePoints(const cv::Mat& grayImage, const std::string& edgeXmlPath);
   // bool loadEdgePointsAndGenerateTemplates(const std::string& edgeXmlPath);

    // 匹配接口
    bool matchAngleTemplate(const cv::Mat& grayImage, std::vector<cv::Mat>& results);
    bool matchFttTemplate(const cv::Mat& img, std::vector<cv::Mat>& tempdft, cv::Size corrsize, std::vector<cv::Mat>& result, const int& method);
    bool matchNCCTemplate(const cv::Mat& img, const std::vector<cv::Mat>& templ, std::vector<cv::Mat>& result, const int& method);
    bool findBestMatch(const std::vector<cv::Mat>& results, cv::Point2f& bestLoc,
        float& bestAngle, double& bestScore);
    bool  matchLocation(cv::Mat& image, const int& method, std::vector<std::pair<double, cv::Point2f>>& location, int x, int y, double& val);

    /**
 * @brief 完整匹配流程（从XML加载边缘点→生成模板→匹配→解析结果）
 * @param testImagePath 待匹配图像路径
 * @param edgeXmlPath 边缘点XML文件路径（用于加载边缘点并生成模板）
 * @param bestLoc[out] 最佳匹配位置
 * @param bestAngle[out] 最佳匹配角度（弧度）
 * @param bestScore[out] 最佳匹配得分
 * @return 是否成功
 */
    //bool runFullMatchingFromPath(
    //    const std::string& testImagePath,
    //    const std::string& edgeXmlPath,  // 新增：边缘点XML路径
    //    cv::Point2f& bestLoc,
    //    float& bestAngle,
    //    double& bestScore
    //);
    bool runFullMatchingFromPath(
        const cv::Mat& grayImage,
        const std::string& edgeXmlPath,  // 新增：边缘点XML路径
        cv::Point2f& bestLoc,
        float& bestAngle,
        double& bestScore
    );
    
    /**
     * @brief 从已加载的灰度图执行完整匹配流程（匹配→解析结果）
     * @param testGray 已加载的单通道灰度图（CV_8UC1）
     * @param bestLoc[out] 最佳匹配中心坐标
     * @param bestAngle[out] 最佳匹配角度（弧度制）
     * @param bestScore[out] 最佳匹配得分（0~1，越高越优）
     * @return 是否成功（true=成功找到有效匹配，false=失败或无匹配）
     */
    bool runFullMatchingFromGrayMat(
        const cv::Mat& testGray,
        cv::Point2f& bestLoc,
        float& bestAngle,
        double& bestScore
    );
    double cannyWithOpenCL(const cv::Mat& input, cv::Mat& output, double threshold1, double threshold2);
    double cannyWithoutOpenCL(const cv::Mat& input, cv::Mat& output, double threshold1, double threshold2);
    double processWithPossibleIPP(const cv::Mat& input, cv::Mat& output);
    double processWithoutIPP(const cv::Mat& input, cv::Mat& output);


    // 原有对外接口
    size_t getTemplateCount() const { return spatialTemplates_.size(); }
    std::vector<cv::Mat> getFourierTemplates() const { return fourierTemplates_; }
    cv::Mat getLastEdgeImage() const { return lastEdgeImage_; }
    /////////////////////////////////////
    bool Region_test(
        const cv::Mat& grayImage,
        const std::string& edgeXmlPath,  // 新增：边缘点XML路径
        cv::Point2f& bestLoc,
        float& bestAngle,
        double& bestScore
    );

    bool Region_test_subpix(
        const cv::Mat& grayImage,
        const std::string& edgeXmlPath,  // 新增：边缘点XML路径
        cv::Point2f& bestLoc,
        float& bestAngle,
        double& bestScore
    );

};
