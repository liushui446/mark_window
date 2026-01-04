// precise_offset_solver.cpp
// 用于图像中基于旋转和平移参数的精确偏移量拟合计算

#include "precise_offset_solver.h"
#include <opencv2/opencv.hpp>
#include <cmath>

#define AToR CV_PI/180 
int precise_offset_solver::findPreciseOffset(const cv::Mat& image, std::vector<cv::Point2f>& points, double& r, double& x, double& y)
{
    cv::Mat params(3, 1, CV_64F);
    params.at<double>(0, 0) = r;
    params.at<double>(1, 0) = x;
    params.at<double>(2, 0) = y;

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 1e-4);
    int max_iter = 50;

    cv::Mat err, jac;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        if (calcJacobian(jac, image, points, params.at<double>(0, 0), params.at<double>(1, 0), params.at<double>(2, 0))) return 1;
        if (calcError(err, image, points, params.at<double>(0, 0), params.at<double>(1, 0), params.at<double>(2, 0))) return 1;

        cv::Mat jt = jac.t();
        cv::Mat jtj = jt * jac;
        cv::Mat jte = jt * err;

        cv::Mat delta;
        cv::solve(jtj, jte, delta, cv::DECOMP_SVD);

        params -= delta;

        if (cv::norm(delta) < 1e-4)
            break;
    }

    r = params.at<double>(0, 0);
    x = params.at<double>(1, 0);
    y = params.at<double>(2, 0);

    return 0;
}
//int precise_offset_solver::findPreciseOffset(const cv::Mat& image, std::vector<cv::Point2f>& points, double& r, double& x, double& y)
//{
//    cv::Mat params(3, 1, CV_64F);
//    params.at<double>(0, 0) = r;
//    params.at<double>(1, 0) = x;
//    params.at<double>(2, 0) = y;
//
//    const int max_iter = 50;
//    const double epsilon = 1e-5;
//    double lambda = 1e-3;  // LM 阻尼初值
//
//    cv::Mat err, jac;
//    double last_error_norm = std::numeric_limits<double>::max();
//
//    for (int iter = 0; iter < max_iter; ++iter)
//    {
//        if (calcJacobian(jac, image, points, params.at<double>(0, 0), params.at<double>(1, 0), params.at<double>(2, 0))) return 1;
//        if (calcError(err, image, points, params.at<double>(0, 0), params.at<double>(1, 0), params.at<double>(2, 0))) return 1;
//
//        cv::Mat jt = jac.t();
//        cv::Mat jtj = jt * jac;
//        cv::Mat jte = jt * err;
//
//        // 添加阻尼项: LM核心
//        cv::Mat lm_matrix = jtj + lambda * cv::Mat::eye(jtj.size(), jtj.type());
//
//        cv::Mat delta;
//        cv::solve(lm_matrix, jte, delta, cv::DECOMP_SVD);
//
//        // 尝试更新
//        cv::Mat new_params = params - delta;
//
//        // 计算新误差
//        cv::Mat new_err;
//        if (calcError(new_err, image, points, new_params.at<double>(0, 0), new_params.at<double>(1, 0), new_params.at<double>(2, 0))) return 1;
//
//        double new_error_norm = cv::norm(new_err);
//
//        if (new_error_norm < last_error_norm)
//        {
//            // 收敛更好了，接受更新，减少阻尼
//            params = new_params;
//            last_error_norm = new_error_norm;
//            lambda *= 0.7;
//        }
//        else
//        {
//            // 收敛变差了，增加阻尼，放弃这一步
//            lambda *= 2.0;
//        }
//
//        if (cv::norm(delta) < epsilon)
//            break;
//    }
//
//    r = params.at<double>(0, 0);
//    x = params.at<double>(1, 0);
//    y = params.at<double>(2, 0);
//
//    return 0;
//}

// 使用 FLANN 构建 K-D 树并查找最近邻点
std::vector<cv::Point2f> findNearestNeighbors(
    const std::vector<cv::Point2f>& modelPoints,
    const std::vector<cv::Point2f>& dataPoints)
{
    cv::Mat modelMat(modelPoints.size(), 2, CV_32F);
    for (size_t i = 0; i < modelPoints.size(); ++i) {
        modelMat.at<float>(i, 0) = modelPoints[i].x;
        modelMat.at<float>(i, 1) = modelPoints[i].y;
    }

    cv::flann::Index kdtree(modelMat, cv::flann::KDTreeIndexParams(1));
    std::vector<cv::Point2f> matched;

    for (const auto& pt : dataPoints) {
        cv::Mat query = (cv::Mat_<float>(1, 2) << pt.x, pt.y);
        std::vector<int> indices(1);
        std::vector<float> dists(1);
        kdtree.knnSearch(query, indices, dists, 1, cv::flann::SearchParams(32));
        matched.push_back(modelPoints[indices[0]]);
    }

    return matched;
}

// SVD 计算刚性变换（旋转 + 平移）
cv::Mat estimateRigidTransformSVD(
    const std::vector<cv::Point2f>& src,
    const std::vector<cv::Point2f>& dst,
    double& theta, double& dx, double& dy)
{
    cv::Point2f src_centroid(0, 0), dst_centroid(0, 0);
    for (size_t i = 0; i < src.size(); ++i) {
        src_centroid += src[i];
        dst_centroid += dst[i];
    }
    src_centroid *= 1.0 / src.size();
    dst_centroid *= 1.0 / dst.size();

    std::vector<cv::Point2f> src_demean, dst_demean;
    for (size_t i = 0; i < src.size(); ++i) {
        src_demean.push_back(src[i] - src_centroid);
        dst_demean.push_back(dst[i] - dst_centroid);
    }

    cv::Mat H = cv::Mat::zeros(2, 2, CV_64F);
    for (size_t i = 0; i < src.size(); ++i) {
        H.at<double>(0, 0) += src_demean[i].x * dst_demean[i].x;
        H.at<double>(0, 1) += src_demean[i].x * dst_demean[i].y;
        H.at<double>(1, 0) += src_demean[i].y * dst_demean[i].x;
        H.at<double>(1, 1) += src_demean[i].y * dst_demean[i].y;
    }

    cv::Mat w, u, vt;
    cv::SVD::compute(H, w, u, vt);
    cv::Mat R = vt.t() * u.t();

    if (cv::determinant(R) < 0) {
        vt.row(1) *= -1;
        R = vt.t() * u.t();
    }

    cv::Mat t = (cv::Mat_<double>(2, 1) << dst_centroid.x, dst_centroid.y)
        - R * (cv::Mat_<double>(2, 1) << src_centroid.x, src_centroid.y);

    dx = t.at<double>(0, 0);
    dy = t.at<double>(1, 0);
    theta = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));

    // 返回仿射变换矩阵
    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 2, 2)));
    T.at<double>(0, 2) = dx;
    T.at<double>(1, 2) = dy;
    return T;
}
int precise_offset_solver::findRigidTransform2D(const std::vector<cv::Point2f>& modelPoints,
    const std::vector<cv::Point2f>& dataPoints,
    double& theta, double& dx, double& dy,
    int max_iter , double epsilon )
{
    if (modelPoints.empty() || dataPoints.empty())
        return -1;
    double theta1 = 0.0;
    double dx1 = 0.0;
    double dy1 = 0.0;
    std::vector<cv::Point2f> transformed = dataPoints;
    //theta = 0.0;
    //dx = 0.0;
    //dy = 0.0;
    std::vector<cv::Point2f> modelShifted;
    modelShifted.reserve(modelPoints.size());
    for (const auto& pt : modelPoints) {
        modelShifted.emplace_back(pt.x + dx, pt.y + dy);
    }
    // ---- 构建 K-D Tree（只对 modelPoints 建一次）----
    cv::Mat modelMat(modelPoints.size(), 2, CV_32F);
    for (size_t i = 0; i < modelPoints.size(); ++i) {
        modelMat.at<float>(i, 0) = modelPoints[i].x+ dx;
        modelMat.at<float>(i, 1) = modelPoints[i].y+ dy;
        
    }
    cv::flann::Index kdtree(modelMat, cv::flann::KDTreeIndexParams(1));

    for (int iter = 0; iter < max_iter; ++iter) {
        // ---- Step 1: 动态最近邻匹配 ----
        std::vector<cv::Point2f> matchedModel;
        for (const auto& pt : transformed) {
            cv::Mat query = (cv::Mat_<float>(1, 2) << pt.x, pt.y);
            std::vector<int> indices(1);
            std::vector<float> dists(1);
            kdtree.knnSearch(query, indices, dists, 1, cv::flann::SearchParams());

            matchedModel.push_back(modelShifted[indices[0]]);
        }

        // ---- Step 2: 计算质心 ----
        cv::Point2f centroidModel(0, 0), centroidData(0, 0);
        for (size_t i = 0; i < transformed.size(); ++i) {
            centroidModel += matchedModel[i];
            centroidData += transformed[i];
        }
        centroidModel *= (1.0 / transformed.size());
        centroidData *= (1.0 / transformed.size());

        // ---- Step 3: 去中心化 ----
        std::vector<cv::Point2f> q, p;
        for (size_t i = 0; i < transformed.size(); ++i) {
            q.push_back(matchedModel[i] - centroidModel);  // 模板点
            p.push_back(transformed[i] - centroidData);    // 当前点
        }

        // ---- Step 4: 计算协方差矩阵 ----
        cv::Mat H = cv::Mat::zeros(2, 2, CV_64F);
        for (size_t i = 0; i < p.size(); ++i) {
            H.at<double>(0, 0) += p[i].x * q[i].x;
            H.at<double>(0, 1) += p[i].x * q[i].y;
            H.at<double>(1, 0) += p[i].y * q[i].x;
            H.at<double>(1, 1) += p[i].y * q[i].y;
        }

        // ---- Step 5: SVD 得到旋转 ----
        cv::Mat U, S, Vt;
        cv::SVD::compute(H, S, U, Vt);
        cv::Mat R = Vt.t() * U.t();
        if (cv::determinant(R) < 0) {
            Vt.row(1) *= -1;
            R = Vt.t() * U.t();
        }

        double dtheta = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));

        // ---- Step 6: 计算平移 ----
        cv::Mat Rc = (cv::Mat_<double>(2, 1) << centroidData.x, centroidData.y);
        cv::Mat Rc_rot = R * Rc;
        cv::Point2f delta = centroidModel - cv::Point2f(Rc_rot.at<double>(0), Rc_rot.at<double>(1));

        // ---- Step 7: 应用变换 ----
        for (auto& pt : transformed) {
            double x_new = std::cos(dtheta) * pt.x - std::sin(dtheta) * pt.y + delta.x;
            double y_new = std::sin(dtheta) * pt.x + std::cos(dtheta) * pt.y + delta.y;
            pt = cv::Point2f(x_new, y_new);
        }

        // ---- Step 8: 收敛判断 ----
        if (std::abs(dtheta) < epsilon && cv::norm(delta) < epsilon)
            break;

        theta1 += dtheta;
        dx1 += delta.x;
        dy1 += delta.y;
    }
    theta -= theta1;
    dx -= dx1;
    dy -= dy1;
    return 0;
}
int precise_offset_solver::calcJacobian(cv::Mat& jac, const cv::Mat& image, std::vector<cv::Point2f>& points, double r, double x, double y)
{
    jac.create(points.size(), 3, CV_64F);

    double step = 0.1;
    cv::Mat err1_, err2_;

    if (calcError(err1_, image, points, r - step, x, y)) return 1;
    if (calcError(err2_, image, points, r + step, x, y)) return 1;

    cv::Mat col0 = jac.col(0);
    calcDeriv(err1_, err2_, 2 * step, col0);

    step = 0.1;
    if (calcError(err1_, image, points, r, x - step, y)) return 1;
    if (calcError(err2_, image, points, r, x + step, y)) return 1;

    cv::Mat col1 = jac.col(1);
    calcDeriv(err1_, err2_, 2 * step, col1);

    if (calcError(err1_, image, points, r, x, y - step)) return 1;
    if (calcError(err2_, image, points, r, x, y + step)) return 1;

    cv::Mat col2 = jac.col(2);
    calcDeriv(err1_, err2_, 2 * step, col2);

    return 0;
}
float bilinearInterpolation(const cv::Mat& img, cv::Point2f pt) {
    int x = int(pt.x), y = int(pt.y);
    if (x < 0 || y < 0 || x >= img.cols - 1 || y >= img.rows - 1) return 0.0f;

    float dx = pt.x - x, dy = pt.y - y;
    float a = img.at<float>(y, x);
    float b = img.at<float>(y, x + 1);
    float c = img.at<float>(y + 1, x);
    float d = img.at<float>(y + 1, x + 1);

    return (1 - dx) * (1 - dy) * a + dx * (1 - dy) * b + (1 - dx) * dy * c + dx * dy * d;
}
int precise_offset_solver::calcError(cv::Mat& err, const cv::Mat& image, std::vector<cv::Point2f>& points, double r, double x, double y)
{
    err.create(points.size(), 1, CV_64F);
    for (unsigned int i = 0; i < points.size(); i++)
    {
        cv::Point2f transpoint;
        //transpoint.x = cvRound(points[i].x * cos(r * AToR) -
        //    points[i].y * sin(r * AToR) + x);
        //transpoint.y = cvRound(points[i].x * sin(r * AToR) +
        //    points[i].y * cos(r * AToR) + y);
        transpoint.x = points[i].x * cos(r * AToR) -
            points[i].y * sin(r * AToR) + x;
        transpoint.y = points[i].x * sin(r * AToR) +
            points[i].y * cos(r * AToR) + y;

        if (transpoint.x < 0 || transpoint.x >= image.cols || transpoint.y < 0 || transpoint.y >= image.rows)
        {
            return 1;
        }
        float val = bilinearInterpolation(image, transpoint);
        err.at<double>(i, 0) = val;  // 如果你期望最小值
        //err.at<double>(i, 0) = image.at<float>(transpoint.y, transpoint.x);
    }
    return 0;
}

void precise_offset_solver::calcDeriv(const cv::Mat& err1, const cv::Mat& err2, double h, cv::Mat& res)
{
    for (int i = 0; i < err1.rows; ++i)
        res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}
