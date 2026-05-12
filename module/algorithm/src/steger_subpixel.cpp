#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include "StegerSubpixel.h"
using namespace cv;
using namespace std;
const double PI = 3.14159265358979323846;
// 定义3x3和5x5邻域的相对坐标
const int R_3x3[3] = { -1, 0, 1 };
const int C_3x3[3] = { -1, 0, 1 };
const int R_5x5[5] = { -2, -1, 0, 1, 2 };
const int C_5x5[5] = { -2, -1, 0, 1, 2 };

// 定义10个正交多项式基函数
double P0(int r, int c) { return 1.0; }
double P1(int r, int c) { return r; }
double P2(int r, int c) { return c; }
double P3(int r, int c) { return r * r - 2; }
double P4(int r, int c) { return r * c; }
double P5(int r, int c) { return c * c - 2; }
double P6(int r, int c) { return r * r * r - (17.0 / 5.0) * r; }
double P7(int r, int c) { return (r * r - 2) * c; }
double P8(int r, int c) { return r * (c * c - 2); }
double P9(int r, int c) { return c * c * c - (17.0 / 5.0) * c; }

// 基函数数组
typedef double (*BasisFunc)(int, int);
const BasisFunc basisFuncs[10] = { P0, P1, P2, P3, P4, P5, P6, P7, P8, P9 };

// 预计算基函数范数
vector<double> precomputeBasisNorms(int size) {
    vector<double> norms(10, 0.0);
    const int* R;
    const int* C;
    int dim;

    if (size == 3) {
        R = R_3x3;
        C = C_3x3;
        dim = 3;
    }
    else if (size == 5) {
        R = R_5x5;
        C = C_5x5;
        dim = 5;
    }
    else {
        return norms;
    }

    for (int i = 0; i < 10; ++i) {
        double norm = 0.0;
        for (int r_idx = 0; r_idx < dim; ++r_idx) {
            for (int c_idx = 0; c_idx < dim; ++c_idx) {
                int r = R[r_idx];
                int c = C[c_idx];
                double val = basisFuncs[i](r, c);
                norm += val * val;
            }
        }
        norms[i] = norm;
    }
    return norms;
}

// 全局范数（预计算3x3和5x5窗口的基函数范数）
const vector<double> basisNorms_3x3 = precomputeBasisNorms(3);
const vector<double> basisNorms_5x5 = precomputeBasisNorms(5);

// 拟合Facet模型（修复引用初始化问题）
vector<double> fitFacetModel(const Mat& window) {
    if (window.type() != CV_32F || (window.rows != 3 || window.cols != 3) && (window.rows != 5 || window.cols != 5)) {
        return {}; // 仅支持3x3或5x5的CV_32F矩阵
    }

    int size = window.rows;
    const int* R;
    const int* C;
    const vector<double>* basisNorms; // 使用指针替代引用
    int dim;

    // 根据窗口尺寸选择参数
    if (size == 3) {
        R = R_3x3;
        C = C_3x3;
        basisNorms = &basisNorms_3x3; // 指向3x3的范数
        dim = 3;
    }
    else { // size == 5
        R = R_5x5;
        C = C_5x5;
        basisNorms = &basisNorms_5x5; // 指向5x5的范数
        dim = 5;
    }

    vector<double> coeffs(10, 0.0);
    for (int i = 0; i < 10; ++i) {
        BasisFunc func = basisFuncs[i];
        double numerator = 0.0;
        for (int r_idx = 0; r_idx < dim; ++r_idx) {
            for (int c_idx = 0; c_idx < dim; ++c_idx) {
                int r = R[r_idx];
                int c = C[c_idx];
                double I_val = window.at<float>(r_idx, c_idx);
                double Pi_val = func(r, c);
                numerator += I_val * Pi_val;
            }
        }
        // 通过指针访问范数值
        double denominator = (*basisNorms)[i];
        coeffs[i] = (fabs(denominator) < 1e-6) ? 0.0 : (numerator / denominator);
    }
    return coeffs;
}

// 计算导数和Hessian矩阵
bool computeDerivatives(const vector<double>& coeffs, int windowSize,
    double& Ix, double& Iy, double& Ixx, double& Ixy, double& Iyy) {
    if (coeffs.size() != 10) return false;

    Ix = coeffs[1];  // P1对应x方向一阶导数
    Iy = coeffs[2];  // P2对应y方向一阶导数
    Ixx = 2 * coeffs[3];  // P3对应x方向二阶导数
    Ixy = coeffs[4];      // P4对应混合二阶导数
    Iyy = 2 * coeffs[5];  // P5对应y方向二阶导数

    return true;
}

// 提取初始边缘点
vector<Point> extractEdgePoints(const Mat& grayImage, double threshold1 = 50, double threshold2 = 150, int skip = 1) {
    vector<Point> edgePoints;
    if (grayImage.empty() || grayImage.channels() != 1) return edgePoints;

    Mat blurred, edges;
    GaussianBlur(grayImage, blurred, Size(3, 3), 1.0);
    Canny(blurred, edges, threshold1, threshold2, 3, true);

    for (int y = 0; y < edges.rows; y += skip) {
        for (int x = 0; x < edges.cols; x += skip) {
            if (edges.at<uchar>(y, x) > 0) {
                edgePoints.push_back(Point(x, y));
            }
        }
    }
    return edgePoints;
}

// 亚像素边缘提取核心函数
vector<Point2f> facetHessianSubpixel(const Mat& image, const vector<Point>& edgePoints) {
    vector<Point2f> subpixelPoints;
    if (image.empty() || edgePoints.empty()) return subpixelPoints;

    Mat gray, floatImg;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = image.clone();
    }
    gray.convertTo(floatImg, CV_32F);
    GaussianBlur(floatImg, floatImg, Size(3, 3), 1.0);

    const double MAX_OFFSET = 0.5;
    const double MIN_D = 1e-3;
    const double LAMBDA_RATIO = 0.5;

    for (const Point& pt : edgePoints) {
        bool valid = false;
        double Ix, Iy, Ixx, Ixy, Iyy;
        double t = 0.0;
        double nx = 0.0, ny = 0.0;

        // 尝试3x3窗口
        if (pt.x >= 1 && pt.x < image.cols - 1 && pt.y >= 1 && pt.y < image.rows - 1) {
            Rect roi3x3(pt.x - 1, pt.y - 1, 3, 3);
            Mat window3x3 = floatImg(roi3x3);
            vector<double> coeffs3x3 = fitFacetModel(window3x3);

            if (!coeffs3x3.empty() && computeDerivatives(coeffs3x3, 3, Ix, Iy, Ixx, Ixy, Iyy)) {
                double trace = Ixx + Iyy;
                double det = Ixx * Iyy - Ixy * Ixy;

                if (det > 0) {
                    double sqrtVal = sqrt(trace * trace / 4.0 - det);
                    double lambda1 = trace / 2.0 - sqrtVal;
                    double lambda2 = trace / 2.0 + sqrtVal;

                    if (fabs(lambda1) <= LAMBDA_RATIO * fabs(lambda2)) {
                        if (fabs(Ixy) > 1e-6) {
                            nx = lambda1 - Iyy;
                            ny = Ixy;
                        }
                        else {
                            nx = 0.0;
                            ny = 1.0;
                        }
                        double norm = sqrt(nx * nx + ny * ny);
                        if (norm > 1e-6) {
                            nx /= norm;
                            ny /= norm;

                            double D = nx * nx * Ixx + 2 * nx * ny * Ixy + ny * ny * Iyy;
                            if (fabs(D) < MIN_D) D = (D > 0) ? MIN_D : -MIN_D;
                            t = -(nx * Ix + ny * Iy) / D;

                            if (fabs(t) <= MAX_OFFSET) {
                                valid = true;
                            }
                        }
                    }
                }
            }
        }

        // 3x3无效则尝试5x5窗口
        if (!valid) {
            if (pt.x >= 2 && pt.x < image.cols - 2 && pt.y >= 2 && pt.y < image.rows - 2) {
                Rect roi5x5(pt.x - 2, pt.y - 2, 5, 5);
                Mat window5x5 = floatImg(roi5x5);
                vector<double> coeffs5x5 = fitFacetModel(window5x5);

                if (!coeffs5x5.empty() && computeDerivatives(coeffs5x5, 5, Ix, Iy, Ixx, Ixy, Iyy)) {
                    double trace = Ixx + Iyy;
                    double det = Ixx * Iyy - Ixy * Ixy;

                    if (det > 0) {
                        double sqrtVal = sqrt(trace * trace / 4.0 - det);
                        double lambda1 = trace / 2.0 - sqrtVal;
                        double lambda2 = trace / 2.0 + sqrtVal;

                        if (fabs(lambda1) <= LAMBDA_RATIO * fabs(lambda2)) {
                            if (fabs(Ixy) > 1e-6) {
                                nx = lambda1 - Iyy;
                                ny = Ixy;
                            }
                            else {
                                nx = 0.0;
                                ny = 1.0;
                            }
                            double norm = sqrt(nx * nx + ny * ny);
                            if (norm > 1e-6) {
                                nx /= norm;
                                ny /= norm;

                                double D = nx * nx * Ixx + 2 * nx * ny * Ixy + ny * ny * Iyy;
                                if (fabs(D) < MIN_D) D = (D > 0) ? MIN_D : -MIN_D;
                                t = -(nx * Ix + ny * Iy) / D;

                                if (fabs(t) > MAX_OFFSET) {
                                    //t = 0;
                                    t = (t > 0) ? MAX_OFFSET : -MAX_OFFSET;
                                }
                                valid = true;
                            }
                        }
                    }
                }
            }
        }

        // 计算最终坐标
        if (valid) {
            float subX = pt.x + static_cast<float>(t * nx);
            float subY = pt.y + static_cast<float>(t * ny);
            subpixelPoints.push_back(Point2f(subX, subY));
        }
        else {
            subpixelPoints.push_back(Point2f(static_cast<float>(pt.x) + 0.01f,
                static_cast<float>(pt.y) + 0.01f));
        }
    }

    // 保存结果
    ofstream outFile("sub_edge_points.txt");
    for (const Point2f& p : subpixelPoints) {
        outFile << p.x << " " << p.y << endl;
    }
    outFile.close();

    return subpixelPoints;
}
int SubPixelByZernike1(const cv::Mat& src,
    const std::vector<cv::Point>& vecFinalKeypoints,
    std::vector<cv::Point2f>& group_pin_tough_corners)
{
    constexpr int    nbsize = 5;
    constexpr int    halfN = (nbsize - 1) / 2;          // = 2
    constexpr double scaleR = (nbsize - 1) / 2.0;        // = 2.0  ★ 关键：用 (N-1)/2，不是 N/2
    constexpr double L_MAX = 1.0;                        // |l| 上限：超过 → 拒绝
    constexpr double M11_MIN = 1.0;                        // |M11| 下限：太小 → 无边
    constexpr double M00_FLAT_TOL = 1e-3;                  // 平区检测（按你的归一化调）

    // ---- 1) 轻度高斯模糊，抑制噪声对矩的污染 ----
    cv::Mat smooth;
    cv::GaussianBlur(src, smooth, cv::Size(3, 3), 0.8);

    const int W = smooth.cols;
    const int H = smooth.rows;

    group_pin_tough_corners.clear();
    group_pin_tough_corners.resize(vecFinalKeypoints.size());

    // ---- 2) 并行 + 拒绝无效解 ----
    std::vector<uchar> valid(vecFinalKeypoints.size(), 0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)vecFinalKeypoints.size(); ++i) {
        const cv::Point& kp = vecFinalKeypoints[i];

        // 边界保护：邻域必须完整落在图像内
        if (kp.x < halfN || kp.y < halfN ||
            kp.x >= W - halfN || kp.y >= H - halfN) {
            group_pin_tough_corners[i] = cv::Point2f(kp.x, kp.y);  // 退化：原点
            continue;
        }

        // 取 5×5 邻域（无 adjustROI 隐患）
        cv::Mat neibor = smooth(cv::Rect(kp.x - halfN, kp.y - halfN, nbsize, nbsize));

        double M00 = 0, M11R = 0, M11I = 0, M20 = 0;
        CalulateCon(neibor, ZERPOLY00, M00);
        CalulateCon(neibor, ZERPOLY11R, M11R);
        CalulateCon(neibor, ZERPOLY11I, M11I);
        CalulateCon(neibor, ZERPOLY20, M20);

        // 边方向
        const double phi = std::atan2(M11I, M11R);
        const double cphi = std::cos(phi);
        const double sphi = std::sin(phi);

        // 旋转后的 M11（取实部即为沿边法向的幅值）
        const double rM11 = cphi * M11R + sphi * M11I;

        // ---- 拒绝无效解：邻域无真实边 ----
        if (std::abs(rM11) < M11_MIN) {
            group_pin_tough_corners[i] = cv::Point2f(kp.x, kp.y);
            continue;
        }

        // 解出归一化距离 l
        const double l = M20 / rM11;

        // ---- 拒绝模型外样本：|l| > 1 表示边不在圆盘内 ----
        if (!std::isfinite(l) || std::abs(l) > L_MAX) {
            group_pin_tough_corners[i] = cv::Point2f(kp.x, kp.y);
            continue;
        }

        // ---- 偏移：注意尺度因子用 (N-1)/2 而非 N/2 ----
        // y 的正负号取决于 ZERPOLY11I 的定义，下面的 "+sin" 是
        // 假设 ZERPOLY11I 用图像坐标（y 向下）。若你的多项式是数学
        // 坐标（y 向上），把 + 改回 -。验证方法：见正文最后一节。
        const float dx = static_cast<float>(l * scaleR * cphi);
        const float dy = static_cast<float>(l * scaleR * sphi);

        group_pin_tough_corners[i] = cv::Point2f(
            static_cast<float>(kp.x) + dx,
            static_cast<float>(kp.y) + dy        // ← 若结果反了改成 -dy
        );
        valid[i] = 1;
    }

    // ---- 3) 紧凑化：剔除被拒绝的点（可选）----
    // 如果调用方需要"边缘点和亚像素点一一对应"，就保留 valid==0 的点用原坐标；
    // 如果需要"高质量子集"，下面这段把无效点剔除。
    /*
    size_t k = 0;
    for (size_t i = 0; i < group_pin_tough_corners.size(); ++i) {
        if (valid[i]) group_pin_tough_corners[k++] = group_pin_tough_corners[i];
    }
    group_pin_tough_corners.resize(k);
    */

    return 0;
}
int SubPixelByZernike(cv::Mat src, std::vector<cv::Point> vecFinalKeypoints, std::vector<cv::Point2f>& group_pin_tough_corners)
{
    int nbsize = 5;
    Mat matInRoi, matNewRoi, matNeibor;
    int halfN = (nbsize - 1) / 2;
    for (int i = 0; i < vecFinalKeypoints.size(); i++)
    {
        matInRoi = src(Rect(vecFinalKeypoints[i].x, vecFinalKeypoints[i].y, 1, 1));
        matInRoi.adjustROI(halfN, halfN, halfN, halfN);
        matNeibor = matInRoi.clone();
#if image_show
        Mat showImage;
        cvtColor(src, showImage, COLOR_GRAY2BGR);
        circle(showImage, (cv::Point)vecFinalKeypoints[i], 5, cv::Scalar(0, 255, 0));

#endif
        Mat showImage;
        cvtColor(src, showImage, COLOR_GRAY2BGR);
        circle(showImage, (cv::Point)vecFinalKeypoints[i], 5, cv::Scalar(0, 255, 0));
        double phi, l, k, h;
        double M00, M11R, M11I, M20;
        CalulateCon(matNeibor, ZERPOLY00, M00);
        CalulateCon(matNeibor, ZERPOLY11R, M11R);
        CalulateCon(matNeibor, ZERPOLY11I, M11I);
        CalulateCon(matNeibor, ZERPOLY20, M20);
        double rM00, rM11R, rM11I, rM20, rM11;
        rM00 = M00;
        phi = atan2(M11I, M11R);
        rM11R = cos(phi) * M11R + sin(phi) * M11I;
        rM11I = cos(phi) * M11I - sin(phi) * M11R;
        rM11 = rM11R;
        rM20 = M20;

        l = M20 / rM11;
        k = 3 * rM11 / (2 * pow((1 - l * l), 1.5));
        h = (M00 - k * PI / 2 + k * sin(l) + k * l * sqrt(1 - l * l)) / PI;
        cv::Point2f subPoint;
        subPoint.x = (vecFinalKeypoints[i].x + l * nbsize * cos(phi) / 2.00);
        subPoint.y = (vecFinalKeypoints[i].y - l * nbsize * sin(phi) / 2.00);
        //subPoint.y = (vecFinalKeypoints[i].y + l * nbsize * sin(phi) / 2.00);
        group_pin_tough_corners.push_back(subPoint);
    }

    Mat showImage;
    cvtColor(src, showImage, COLOR_GRAY2BGR);
    for (unsigned int i = 0; i < group_pin_tough_corners.size(); i++)
    {
        //std::cout<<group_pin_tough_corners[i]<<std::endl;
        circle(showImage, (cv::Point)group_pin_tough_corners[i], 2, cv::Scalar(0, 255, 0));
        circle(showImage, (cv::Point)vecFinalKeypoints[i], 2, cv::Scalar(255, 0, 0));
    }
#if image_show
    Mat showImage;
    cvtColor(src, showImage, COLOR_GRAY2BGR);
    for (unsigned int i = 0; i < group_pin_tough_corners.size(); i++)
    {
        //std::cout<<group_pin_tough_corners[i]<<std::endl;
        circle(showImage, (cv::Point)group_pin_tough_corners[i], 2, cv::Scalar(0, 255, 0));
        circle(showImage, (cv::Point)vecFinalKeypoints[i], 2, cv::Scalar(255, 0, 0));
    }
#endif

#if image_show
    cv::RotatedRect corseRect = cv::minAreaRect(vecFinalKeypoints);
    Point2f vertices[4];
    cv::Mat showImage1 = showImage.clone();
    corseRect.points(vertices);
    for (int i = 0; i < 4; i++)
        line(showImage1, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
#endif


#if image_show
    cv::RotatedRect finalRect = cv::minAreaRect(group_pin_tough_corners);
    Point2f vertices2[4];
    cv::Mat showImage2 = showImage.clone();
    finalRect.points(vertices2);
    for (int i = 0; i < 4; i++)
        line(showImage2, vertices2[i], vertices2[(i + 1) % 4], Scalar(0, 255, 0), 2);
#endif
    return 0;
}
int CalulateCon(cv::Mat src, cv::Mat kernal, double& value)
{
    double sum = 0;
    for (size_t i = 0; i < src.rows; i++)
    {
        for (size_t j = 0; j < src.cols; j++)
        {
            sum += (double)src.at<uchar>(i, j) * kernal.at<double>(i, j);
        }
    }
    value = sum;
    return 0;
}
int SubPixelBySM(cv::Mat src, std::vector<cv::Point> vecFinalKeypoints, std::vector<cv::Point2f>& group_pin_tough_corners)
{
    int nbsize = 5;
    Mat matInRoi, matNewRoi, matNeibor;
    int halfN = (nbsize - 1) / 2;
    std::vector<cv::Point2f>group_pin_tough_corners_zer;
    for (int i = 0; i < vecFinalKeypoints.size(); i++)
    {
        matInRoi = src(Rect(vecFinalKeypoints[i].x, vecFinalKeypoints[i].y, 1, 1));
        matInRoi.adjustROI(halfN, halfN, halfN, halfN);
        matNeibor = matInRoi.clone();
#if image_show
        Mat showImage;
        cvtColor(src, showImage, COLOR_GRAY2BGR);
        circle(showImage, (cv::Point)vecFinalKeypoints[i], 5, cv::Scalar(0, 255, 0));
#endif
        double phi, l, k, h;
        double M00, M01, M10, M11, M02, M20;
        CalulateCon(matNeibor, SM00, M00);
        CalulateCon(matNeibor, SM01, M01);
        CalulateCon(matNeibor, SM10, M10);
        CalulateCon(matNeibor, SM11, M11);
        CalulateCon(matNeibor, SM02, M02);
        CalulateCon(matNeibor, SM20, M20);
        phi = atan2(M01, M10);
        double rM00, rM01, rM10, rM11, rM02, rM20;
        rM00 = M00;
        rM10 = sqrt(M10 * M10 + M01 * M01);
        rM20 = (M10 * M10 * M20 + 2 * M01 * M10 * M11 + M01 * M01 * M02) / (M01 * M01 + M10 * M10);
        rM02 = (M10 * M10 * M02 - 2 * M01 * M10 * M11 + M01 * M01 * M20) / (M01 * M01 + M10 * M10);

        l = (4 * rM20 - rM00) / (3 * rM10);
        k = 3 * rM10 / (2 * sqrt(pow(1 - l * l, 3)));
        h = 1 / (2 * PI) * (2 * rM00 - k * (PI - 2 * asin(l) - 2 * l * sqrt(1 - l * l)));
        cv::Point a = vecFinalKeypoints[i];
        //不可做假数据，不可混淆视听
        cv::Point2f subPoint;
        subPoint.x = (vecFinalKeypoints[i].x + l * nbsize * cos(phi) / 2.00);
        //subPoint.y = (vecFinalKeypoints[i].y - l * nbsize * sin(phi) / 2.00);
        subPoint.y = (vecFinalKeypoints[i].y + l * nbsize * sin(phi) / 2.00);
        group_pin_tough_corners.push_back(subPoint);
    }
    return 0;
}
int SubPixelByOurMethod(cv::Mat src, std::vector<std::vector<cv::Point2i>> vvPoints, std::vector<cv::Point2f>& vvSubPixelPoints)
{
    //根据插值计算亚像素结果
    //输入插值像素点以及邻域像素位置坐标
    //输出亚像素位置
    for (size_t i = 0; i < vvPoints.size(); i++)
    {
        //计算亚像素结果
        std::vector<double> y;
        for (size_t j = 0; j < vvPoints[i].size(); j++)
        {
            y.push_back(src.at<uchar>(vvPoints[i][j].y, vvPoints[i][j].x));
        }
        int idx = floor(vvPoints[i].size() / 2) + 1;
        double l = -(y[idx] - 2 * y[idx + 1] - y[idx + 2]) / (2 * (y[idx] - 3 * y[idx + 1] - 3 * y[idx + 2] - y[idx + 3]));
        cv::Vec4f line_para;
        cv::fitLine(vvPoints[i], line_para, cv::DIST_L2, 0, 0.01, 0.01);
        double A = line_para[1] / line_para[0];
        double B = -1;
        double C = -B * line_para[3] - A * line_para[2];
        double angle = atan(A);// *180 / PI; //-90~90

        cv::Point2f subLoc = vvPoints[i][idx];
        subLoc.x = subLoc.x + l * cos(angle);
        subLoc.x = subLoc.x - l * sin(angle);
        vvSubPixelPoints.push_back(subLoc);
        //std::vector<double> dy = y;
        //std::rotate(y.begin(), y.begin() + 1, y.end());
        //for (size_t j = 0; j< dy.size(); j++)
        //{
        //    dy[i] = abs(y[j] - dy[j]);
        //}
        //std::vector<double> dyy = dy;
        //for (size_t j = 0; j < dyy.size(); j++)
        //{
        //    dyy[i] = abs(dy[j] - dyy[j]);
        //}
        //std::vector<double> dyyy = dyy;
        //for (size_t j = 0; j < dyyy.size(); j++)
        //{
        //    dyyy[i] = abs(dyy[j] - dyyy[j]);
        //}
    }
    return 0;
}

//bool detectRectangles(const Mat& image, vector<RectangleInfo>& rectInfos, int&minArea ) {
//    if (image.empty()) {
//        cerr << "错误：输入图像为空！" << endl;
//        return false;
//    }
//
//    // 1. 转换为灰度图
//    Mat gray;
//    if (image.channels() == 3) {
//        cvtColor(image, gray, COLOR_BGR2GRAY);
//    }
//    else {
//        gray = image.clone();
//    }
//
//    // 2. 预处理：高斯模糊减少噪声
//    //Mat blurred;
//    //GaussianBlur(gray, blurred, Size(3, 3), 0);
//
//    // 3. 二值化处理
//    Mat binary;
//    threshold(gray, binary, 127, 255, THRESH_BINARY_INV | THRESH_OTSU); // 使用OTSU自动阈值
//    /*adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
//        THRESH_BINARY_INV, 11, 2);*/
//
//    // 4. 形态学操作优化轮廓
//    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
//    morphologyEx(binary, binary, MORPH_CLOSE, kernel); // 填充小空洞
//
//    cv::Mat medianFilteredImage;
//    cv::medianBlur(gray, medianFilteredImage, 3);
//    cv::Mat preprocessedImage;
//    cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度
//
//    // 使用双边滤波减少噪声
//    cv::Mat smoothedImage2, edges2;
//    cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
//    cv::Canny(smoothedImage2, edges2, 170, 200, 3, true);
//    // 5. 轮廓检测：使用RETR_CCOMP获取所有层次轮廓（包括内部轮廓）
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarchy;
//    // RETR_CCOMP模式可以获取所有外层和内层轮廓
//    findContours(edges2, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
//
//    if (contours.empty()) {
//        cerr << "警告：未检测到任何轮廓！" << endl;
//        return true;
//    }
//
//    // 6. 处理轮廓，筛选内部轮廓（重点）
//    rectInfos.clear();
//    for (int i = 0; i < contours.size(); ++i) {
//        // 层次信息说明：hierarchy[i] = [next, previous, child, parent]
//        // 内部轮廓的特征：有父轮廓（parent != -1）且没有子轮廓（child == -1） && hierarchy[i][2] == -1
//        if (hierarchy[i][3] != -1 ) {
//            // 过滤小面积轮廓
//            double area = contourArea(contours[i]);
//            if (area < minArea) {
//                continue;
//            }
//
//            // 轮廓近似为多边形
//            double perimeter = arcLength(contours[i], true);
//            vector<Point> approx;
//            approxPolyDP(contours[i], approx, 0.02 * perimeter, true);
//
//            // 筛选四边形（矩形特征）
//            if (approx.size() != 4) {
//                continue;
//            }
//
//            //// 确保是凸多边形
//            //if (!isContourConvex(approx)) {
//            //    continue;
//            //}
//
//            // 拟合矩形
//            Rect rect = boundingRect(approx);
//
//            // 计算长宽
//            float width = min(rect.width, rect.height);
//            float height = max(rect.width, rect.height);
//
//            // 存储矩形信息
//            RectangleInfo info;
//            info.width = width;
//            info.height = height;
//            info.center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
//            info.boundingRect = rect;
//
//            rectInfos.push_back(info);
//        }
//    }
//
//    //方法2旋转矩形
//    // 定义一个向量用于存储检测到的旋转矩形
//    std::vector<RotatedRect> detectedRects;
//    double maxPerimeter = 0.0;
//    for (size_t i = 0; i < contours.size(); i++) {
//        double perimeter = cv::arcLength(contours[i], true);
//        if (perimeter > maxPerimeter) {
//            maxPerimeter = perimeter;
//        }
//    }
//
//    // 计算最小周长阈值（最大周长的0.1倍）
//    double minPerimeter = 0.2 * maxPerimeter;
//    // 方法2：旋转矩形
//    for (size_t i = 0; i < contours.size(); i++) {
//        // 获取最小外接矩形
//        RotatedRect rotatedRect = minAreaRect(contours[i]);
//        double area = contourArea(contours[i]);
//
//        if (area < minArea) {
//            continue;
//        }
//        double perimeter = cv::arcLength(contours[i], true);
//        if (perimeter < minPerimeter) { // minPerimeter是最小周长阈值
//            continue;
//        }
//
//        // 将矩形存储到向量中
//        detectedRects.push_back(rotatedRect);
//
//        // 提取矩形的四个顶点
//        Point2f vertices[4];
//        rotatedRect.points(vertices);
//
//        // 绘制旋转矩形（使用红色）
//        // 注意：如果gray是单通道灰度图，Scalar只需一个参数；如果是BGR图则需要三个参数(0,0,255)
//        for (int j = 0; j < 4; j++) {
//            // 单通道灰度图用红色（255）
//            line(gray, vertices[j], vertices[(j + 1) % 4], Scalar(255), 1);
//
//            // 如果是BGR彩色图，使用下面这行
//            // line(gray, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 1);
//        }
//    }
//
//    return true;
//}