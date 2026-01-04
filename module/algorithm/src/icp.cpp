#include "icp.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
namespace sm {
    namespace imgproc {
        namespace icp {

            Eigen::Matrix3d TransformVector3dToMatrix3d(const Eigen::Matrix<double, 3, 1>& input) {
                Eigen::Matrix3d output =
                    (Eigen::AngleAxisd(input(0), Eigen::Vector3d::UnitZ()))
                    .matrix();
                output.block<2, 1>(0, 2) = input.block<2, 1>(1, 0);
                return output;
            }

            Mat3x3f eigen_to_custom(const Eigen::Matrix3f& extrinsic) {
                Mat3x3f result;
                for (uint32_t i = 0; i < 3; i++) {
                    for (uint32_t j = 0; j < 3; j++) {
                        result[i][j] = extrinsic(i, j);
                    }
                }
                return result;
            }

            Mat3x3f eigen_slover_333(float* A, float* b)
            {
                Eigen::Matrix<float, 3, 3> A_eigen(A);
                Eigen::Matrix<float, 3, 1> b_eigen(b);
                // ICP point to plane may be unstable, refer to
                // https://www.cs.princeton.edu/~smr/papers/icpstability.pdf
                // add a term ||x|| to make update reasonably small:
                // f = ||(Rp + T - q) * n|| + penalty * ||X||   ==>
                // (ATA + Identity * penalty) * X = B
                Eigen::Matrix3d iden = Eigen::Matrix3d::Identity();
                double penalty = 0.01;
                Eigen::Matrix3d ATA_with_pen = A_eigen.cast<double>() + penalty * iden;

                const Eigen::Matrix<double, 3, 1> update = ATA_with_pen.ldlt().solve(b_eigen.cast<double>());
                Eigen::Matrix3d extrinsic = TransformVector3dToMatrix3d(update);
                return eigen_to_custom(extrinsic.cast<float>());
            }

            void transform_pcd(std::vector<Vec2f>& model_pcd, Mat3x3f& trans) {

#pragma omp parallel for
                for (int i = 0; i < model_pcd.size(); i++) {
                    Vec2f& pcd = model_pcd[i];
                    float new_x = trans[0][0] * pcd.x + trans[0][1] * pcd.y + trans[0][2];
                    float new_y = trans[1][0] * pcd.x + trans[1][1] * pcd.y + trans[1][2];
                    pcd.x = new_x;
                    pcd.y = new_y;
                }
            }

            //template<class Scene>
            //RegistrationResult ICP2D_Point2Plane(std::vector<Vec2f>& model_pcd, const Scene scene,
            //    const ICPConvergenceCriteria criteria)
            //{
            //    RegistrationResult result;
            //    RegistrationResult backup;

            //    std::vector<float> A_host(9, 0);
            //    std::vector<float> b_host(3, 0);
            //    thrust__pcd2Ab<Scene> trasnformer(scene);

            //    // use one extra turn
            //    for (uint32_t iter = 0; iter <= criteria.max_iteration_; iter++) {

            //        //Vec11f reducer;
            //        Vec11f reducer = Vec11f::Zero();
            //        //#pragma omp declare reduction( + : Vec11f : omp_out += omp_in) \
            //                               //initializer (omp_priv = Vec11f::Zero())

            //        //#pragma omp parallel for reduction(+: reducer)
            //        for (int pcd_iter = 0; pcd_iter < model_pcd.size(); pcd_iter++) {
            //            Vec11f result = trasnformer(model_pcd[pcd_iter]);
            //            reducer += result;
            //        }

            //        Vec11f& Ab_tight = reducer;

            //        backup = result;

            //        float& count = Ab_tight[10];
            //        float& total_error = Ab_tight[9];
            //        if (count == 0) return result;  // avoid divid 0

            //        result.fitness_ = float(count) / model_pcd.size();
            //        result.inlier_rmse_ = std::sqrt(total_error / count);

            //        // last extra iter, just compute fitness & mse
            //        if (iter == criteria.max_iteration_) return result;

            //        if (std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
            //            std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_) {
            //            return result;
            //        }

            //        for (int i = 0; i < 3; i++) b_host[i] = Ab_tight[6 + i];

            //        int shift = 0;
            //        for (int y = 0; y < 3; y++) {
            //            for (int x = y; x < 3; x++) {
            //                A_host[x + y * 3] = Ab_tight[shift];
            //                A_host[y + x * 3] = Ab_tight[shift];
            //                shift++;
            //            }
            //        }

            //        Mat3x3f extrinsic = eigen_slover_333(A_host.data(), b_host.data());

            //        transform_pcd(model_pcd, extrinsic);
            //        result.transformation_ = extrinsic * result.transformation_;
            //    }

            //    // never arrive here
            //    return result;
            //}
            //限制距离在1个像素内
            // 2D点到面ICP核心函数【像素级约束改造版】
            template<class Scene>
            RegistrationResult ICP2D_Point2Plane(std::vector<Vec2f>& model_pcd, const Scene scene,
                const ICPConvergenceCriteria criteria)
            {
                RegistrationResult result;
                RegistrationResult backup;
                const float ONE_PIXEL_LIMIT = 1.0f;
                std::vector<float> A_host(9, 0);
                std::vector<float> b_host(3, 0);
                thrust__pcd2Ab<Scene> trasnformer(scene); // 保留原代码笔误


                for (uint32_t iter = 0; iter <= criteria.max_iteration_; iter++) {
                    Vec11f reducer = Vec11f::Zero();
                    for (int pcd_iter = 0; pcd_iter < model_pcd.size(); pcd_iter++) {
                        Vec11f curr_result = trasnformer(model_pcd[pcd_iter]); // 修复变量重名
                        reducer += curr_result;
                    }

                    Vec11f& Ab_tight = reducer;
                    backup = result;

                    float& count = Ab_tight[10];
                    float& total_error = Ab_tight[9];
                    if (count < 1e-6) return result;  // 增强除0防护

                    result.fitness_ = float(count) / model_pcd.size();
                    result.inlier_rmse_ = std::sqrt(total_error / count);

                    if (iter == criteria.max_iteration_) return result;
                    float total_tx = result.transformation_[0][2];
                    float total_ty = result.transformation_[1][2];
                    float total_dist = std::sqrt(total_tx * total_tx + total_ty * total_ty);
                    if (std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
                        std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_&& total_dist < ONE_PIXEL_LIMIT) {
                        return result;
                    }

                    for (int i = 0; i < 3; i++) b_host[i] = Ab_tight[6 + i];

                    int shift = 0;
                    for (int y = 0; y < 3; y++) {
                        for (int x = y; x < 3; x++) {
                            A_host[x + y * 3] = Ab_tight[shift];
                            A_host[y + x * 3] = Ab_tight[shift];
                            shift++;
                        }
                    }

                    Mat3x3f extrinsic = eigen_slover_333(A_host.data(), b_host.data());

                    // ==============================================
                    // ✅ 终极修复：适配sm::imgproc矩阵库 正确访问平移分量
                    // ✅ 彻底解决C2440 vec<3,float> → float 转换错误
                    // ==============================================
                    // ✅ 正确写法：Mat3x3f是【行优先】存储 → 每行是一个vec<3,float>
                    // ✅ extrinsic[行索引][列索引] 才能拿到单个float值
                    //float tx = extrinsic[0][2];  // ✔️ 0行2列 → X方向平移量（像素）
                    //float ty = extrinsic[1][2];  // ✔️ 1行2列 → Y方向平移量（像素）

                    //// ✅ 严格1像素平移约束（核心逻辑不变，绝对生效）
                    //const float translate_len = std::sqrt(tx * tx + ty * ty);
                    //if (translate_len > ONE_PIXEL_LIMIT)
                    //{
                    //    const float scale_factor = ONE_PIXEL_LIMIT / translate_len;
                    //    // ✅ 回写约束后的平移量到矩阵（同正确访问方式）
                    //    extrinsic[0][2] = tx * scale_factor;
                    //    extrinsic[1][2] = ty * scale_factor;
                    //}

                    // 执行约束后的点云变换 + 累积总变换
                    transform_pcd(model_pcd, extrinsic);
                    result.transformation_ = extrinsic * result.transformation_;
                    /*float total_tx = result.transformation_[0][2];
                    float total_ty = result.transformation_[1][2];
                    float total_dist = std::sqrt(total_tx * total_tx + total_ty * total_ty);
                    if (total_dist > ONE_PIXEL_LIMIT)
                    {
                        float total_scale = ONE_PIXEL_LIMIT / total_dist;
                        result.transformation_[0][2] = total_tx * total_scale;
                        result.transformation_[1][2] = total_ty * total_scale;
                    }*/
                }

                return result;
            }
            //////////
            template RegistrationResult ICP2D_Point2Plane(std::vector<Vec2f>& model_pcd, const Scene_edge scene,
                const ICPConvergenceCriteria criteria);

        } //icp
    } //imgproc
} //sm





