#include "KcgMatch.h"
#include "StegerSubpixel.h"
#include <math.h>
#include"facet_model.h"
#include "CannyEdge.h"
#include "precise_offset_solver.h"
#include <fstream>
#include "icp.h"
#include "geometry.h"
//#include <CL/cl.h>
#include <execution>
#define KCG_EPS 0.00001f
#define KCG_PI	3.1415926535897932384626433832795f
#define KCG_MODEL_SUFFUX string(".yaml")
#define M_PI       3.14159265358979323846
const float AngleRegionTable[16][2] = {

	0.f		, 22.5f	,
	22.5f	, 45.f	,
	45.f	, 67.5f	,
	67.5f	, 90.f	,
	90.f	, 112.5f,
	112.5f	, 135.f	,
	135.f	, 157.5f,
	157.5f	, 180.f,
	180.f	, 202.5f,
	202.5f	, 225.f,
	225.f	, 247.5f,
	247.5f	, 270.f,
	270.f	, 292.5f,
	292.5f	, 315.f,
	315.f	, 337.5f,
	337.5f	, 360.f
};

namespace cv_dnn_nms {

	template <typename T>
	static inline bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {

		return pair1.first > pair2.first;
	}

	inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
		std::vector<std::pair<float, int> >& score_index_vec) {

		for (size_t i = 0; i < scores.size(); ++i)
		{
			if (scores[i] > threshold)
			{
				//score_index_vec.push_back(std::make_pair(scores[i], i));
				std::pair<float, int> psi;
				psi.first = scores[i];
				psi.second = (int)i;
				score_index_vec.push_back(psi);
			}
		}
		std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
			SortScorePairDescend<int>);
		if (top_k > 0 && top_k < (int)score_index_vec.size())
		{
			score_index_vec.resize(top_k);
		}
	}

	template <typename BoxType>
	inline void NMSFast_(const std::vector<BoxType>& bboxes,
		const std::vector<float>& scores, const float score_threshold,
		const float nms_threshold, const float eta, const int top_k,
		std::vector<int>& indices, float(*computeOverlap)(const BoxType&, const BoxType&)) {

		CV_Assert(bboxes.size() == scores.size());
		std::vector<std::pair<float, int> > score_index_vec;
		GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

		float adaptive_threshold = nms_threshold;
		indices.clear();
		for (size_t i = 0; i < score_index_vec.size(); ++i) {
			const int idx = score_index_vec[i].second;
			bool keep = true;
			for (int k = 0; k < (int)indices.size() && keep; ++k) {
				const int kept_idx = indices[k];
				float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
				keep = overlap <= adaptive_threshold;
			}
			if (keep)
				indices.push_back(idx);
			if (keep && eta < 1 && adaptive_threshold > 0.5) {
				adaptive_threshold *= eta;
			}
		}
	}

	template<typename _Tp> static inline
		double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
		_Tp Aa = a.area();
		_Tp Ab = b.area();

		if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
			// jaccard_index = 1 -> distance = 0
			return 0.0;
		}

		double Aab = (a & b).area();
		// distance = 1 - jaccard_index
		return 1.0 - Aab / (Aa + Ab - Aab);
	}

	template <typename T>
	static inline float rectOverlap(const T& a, const T& b) {

		return 1.f - static_cast<float>(jaccardDistance__(a, b));
	}

	void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
		const float score_threshold, const float nms_threshold,
		std::vector<int>& indices, const float eta = 1, const int top_k = 0) {

		NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
	}

} // end namespace cv_dnn_nms

namespace kcg {

	KcgMatch::KcgMatch(string model_root, string class_name) {

		assert(!model_root.empty() && "model_root should not empty.");
		assert(!class_name.empty() && "class_name should not empty.");
		if (model_root[model_root.length() - 1] != '/') {

			model_root.push_back('/');
		}
		model_root_ = model_root;
		class_name_ = class_name;

		/// Create 180*180 table
		for (int i = 0; i < 180; i++) {

			for (int j = 0; j < 180; j++) {

				float rad = (i - j) * KCG_PI / 180.f;
				score_table_[i][j] = fabs(cosf(rad));
			}
		}

		/// Create 8*8 table
		ATTR_ALIGN(8) unsigned char score_table_8d[8][8];
		for (int i = 0; i < 8; i++) {

			for (int j = 0; j < 8; j++) {

				float rad = (i - j) * (180.f / 8.f) * KCG_PI / 180.f;
				score_table_8d[i][j] = (unsigned char)(fabs(cosf(rad)) * 100.f);
			}
		}

		/// Create 8*256 table
		for (int i = 0; i < 8; i++) {

			for (int j = 0; j < 256; j++) {

				unsigned char max_score = 0;
				for (int shift_time = 0; shift_time < 8; shift_time++) {

					unsigned char flg = (j >> shift_time) & 0b00000001;
					if (flg) {

						if (score_table_8d[i][shift_time] > max_score) {

							max_score = score_table_8d[i][shift_time];
						}
					}
				}
				score_table_8map_[i][j] = max_score;
			}
		}
	}

	KcgMatch::~KcgMatch() {

	}

//	void KcgMatch::MakingTemplates(Mat model, AngleRange angle_range, ScaleRange scale_range,
//		int num_features, int binarize_thresh, float weak_thresh, float strong_thresh, Mat mask) {
//		ClearModel();
//		std::vector<cv::Point> edgePoints;
//		cv::Mat edge = CannyEdge::detectEdges(model, edgePoints, binarize_thresh);
//		int edgeCount = cv::countNonZero(edge);
//		// 定义边缘点数量阈值（需根据实际场景调试确定）
//		const int THRESH_LOW = 250;    // 边缘点少的阈值
//		const int THRESH_MEDIUM = 500;// 边缘点中的阈值
//		PyramidLevel level;
//		if (edgeCount < THRESH_LOW) {
//			level= PyramidLevel_0;    // 边缘点少，用0级金字塔（原图精度优先）
//		}
//		else if (edgeCount < THRESH_MEDIUM) {
//			level= PyramidLevel_1;    // 边缘点中，用1级金字塔（平衡精度与速度）
//		}
//		else {
//			level= PyramidLevel_2;    // 边缘点多，用2级金字塔（速度优先）
//		}
//		// 寻找边缘点区域的最小边界框
//		//if(edgePoints.size()>0)
//		//{ 
//		//	int minX = edgePoints[0].x, minY = edgePoints[0].y;
//		//	int maxX = edgePoints[0].x, maxY = edgePoints[0].y;
//
//		//	for (size_t i = 1; i < edgePoints.size(); i++) {
//		//		minX = min(minX, edgePoints[i].x);
//		//		minY = min(minY, edgePoints[i].y);
//		//		maxX = max(maxX, edgePoints[i].x);
//		//		maxY = max(maxY, edgePoints[i].y);
//		//	}
//		//	if (minX % 2 == 1) --minX;
//		//	if (minY % 2 == 1) --minY;
//		//	cv::Rect roi_save(minX, minY, maxX - minX + 2, maxY - minY + 2);
//		//	std::string outPath = "D:/mark_window/module/algorithm/src/generate_features.txt";
//		//	std::ofstream outFile(outPath);
//		//	if (!outFile.is_open()) {
//		//		return;
//		//	}
//		//	else {
//		//		for (int i = 0; i < edgePoints.size(); ++i) {
//		//			edgePoints[i].x -= roi_save.x;
//		//			edgePoints[i].y -= roi_save.y;
//		//			outFile << edgePoints[i].x << " " << edgePoints[i].y << endl;
//		//		}
//		//		outFile.close();
//		//	}
//		//	// 裁剪出边缘点区域
//		//	//Rect roi(minX-5, minY-5, maxX - minX + 2+10, maxY - minY + 2+10);
//		//	//Rect roi(minX-60, minY-60, maxX - minX + 120, maxY - minY + 120);
//		//	//Rect roi(528, 396, 150, 148);
//		//	//Rect roi(minX - 2, minY - 2, maxX - minX + 4, maxY - minY + 4 );
//		//	//model = model(roi);
//		//	
//		//}
//
//
//
//		PaddingModelAndMask(model, mask, scale_range.end);
//		angle_range_ = angle_range;
//		scale_range_ = scale_range;
//		vector<ShapeInfo> shape_infos = ProduceShapeInfos(angle_range, scale_range);
//		vector<Mat> l0_mdls; l0_mdls.clear();
//		vector<Mat> l0_msks; l0_msks.clear();
//		for (int s = 0; s < shape_infos.size(); s++) {
//
//			l0_mdls.push_back(MdlOf(model, shape_infos[s]));
//			l0_msks.push_back(MskOf(mask, shape_infos[s]));
//		}
//		for (int p = 0; p <= PyramidLevel_2; p++) {
//
//			for (int s = 0; s < shape_infos.size(); s++) {
//
//				Mat mdl_pyrd = l0_mdls[s];
//				Mat msk_pyrd = l0_msks[s];
//				if (p > 0) {
//
//					Size sz = Size(l0_mdls[s].cols >> 1, l0_mdls[s].rows >> 1);
//					pyrDown(l0_mdls[s], mdl_pyrd, sz);
//					pyrDown(l0_msks[s], msk_pyrd, sz);
//				}
//
//				
//
//				erode(msk_pyrd, msk_pyrd, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
//				l0_mdls[s] = mdl_pyrd;
//				l0_msks[s] = msk_pyrd;
//				// 对于 p = 1，只插入空模板，跳过后续处理
//				//if (p == 1) {
//				//	templ_all_[p].push_back(Template());       // normal 模板占位
//				//	templ_all_[p + 3].push_back(Template());   // gradient flipped 模板占位
//				//	continue;
//				//}
//
//				int features_pyrd = (int)((num_features >> p) * shape_infos[s].scale);
//
//				Mat mag8, angle8, quantized_angle8;
//				edgePoints.clear();
//				
//				//cv::Mat edge2 = CannyEdge::detectEdges(mdl_pyrd, edgePoints, binarize_thresh);
//				cv::Mat edge2;
//				
//				cv::Canny(mdl_pyrd, edge2, 170, 200, 3, true);
//				
//				cv::Mat edge_2;
//				edge_2 = edge2.clone();
//				int shrinkPixels = 4;
//				Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));
//
//				// 2. 腐蚀 mask，缩小白色区域
//				Mat erodedMask;
//				erode(msk_pyrd, erodedMask, kernel);
//
//				// 3. 与 edge 做与操作，去掉缩小区域的边缘
//				bitwise_and(edge_2, erodedMask, edge_2);
//				int whitePixelCount = cv::countNonZero(edge_2);
//				if (whitePixelCount < edge2.cols * 0.1)
//				{
//					cv::Mat medianFilteredImage;
//					cv::medianBlur(mdl_pyrd, medianFilteredImage, 3);
//					cv::Mat preprocessedImage;
//					cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度
//
//					// 使用双边滤波减少噪声
//					cv::Mat smoothedImage2, edges2;
//					cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
//					cv::Canny(smoothedImage2, edge2, 170, 200, 3, true);
//				}
//				vector<vector<Point>> contours;
//				vector<Vec4i> hierarchy; // 存储轮廓层级关系（关键）
//// 使用RETR_CCOMP模式：提取所有轮廓，并分为两级（外部轮廓和内部轮廓）
//// 若需更详细的层级关系，可改用RETR_TREE
//				findContours(edge2, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
//
//				vector<Point> edgePoints;
//				for (const auto& cnt : contours) {
//					if (contourArea(cnt) > 8) {
//						edgePoints.insert(edgePoints.end(), cnt.begin(), cnt.end());
//					}
//				}
//				cv::Mat filtered_edge = cv::Mat::zeros(edge2.size(), edge2.type()); // 创建空白图像（与原边缘图同尺寸）
//
//              // 方法1：直接绘制所有过滤后的点（适合点集较密集的情况）
//				for (const Point& pt : edgePoints) {
//					if (pt.inside(Rect(0, 0, filtered_edge.cols, filtered_edge.rows))) { // 确保点在图像范围内
//						filtered_edge.at<uchar>(pt) = 255; // 边缘点设为白色（255）
//					}
//				}
//				edgePoints.clear();
//				edge2 = filtered_edge.clone();
//				//if (p == 0 && s == 6)
//				//{
//				//	cv::Mat edge3;
//				//	edge3 = edge2.clone();
//				//	int shrinkPixels = 4;
//				//	Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));
//
//				//	// 2. 腐蚀 mask，缩小白色区域
//				//	Mat erodedMask;
//				//	erode(mask, erodedMask, kernel);
//
//				//	// 3. 与 edge 做与操作，去掉缩小区域的边缘
//				//	bitwise_and(edge3, erodedMask, edge3);
//				//	edgePoints.clear();
//				//	for (int y = 0; y < edge3.rows; y++) {
//				//		for (int x = 0; x < edge3.cols; x++) {
//				//			if (edge3.at<uchar>(y, x) > 0) {
//				//				edgePoints.push_back(cv::Point(x, y));
//				//			}
//				//		}
//				//	}
//				//	std::ofstream outFile1("D:/mark_window/module/algorithm/src/edge_points.txt");
//				//	for (const auto& point : edgePoints) {
//				//		outFile1 << point.x << " " << point.y << std::endl;
//				//	}
//				//	outFile1.close();
//				//	FacetEdgeDetector detector(5);
//				//	//使用facet模型进行亚像素边缘检测
//				//	cv::Mat edgeImage = detector.detectEdges(mdl_pyrd, 30.0f);
//				//	// 1. 读取点
//				//	vector<Point2f> subedgePoints;
//				//	ifstream inFile("D:/mark_window/module/algorithm/src/sub_edge_points.txt");
//				//	float x, y;
//				//	while (inFile >> x >> y) {
//				//		subedgePoints.emplace_back(x, y);
//				//	}
//				//	inFile.close();
//
//				//		// 计算裁剪区域
//				//		int min_x = std::numeric_limits<int>::max();
//				//		int min_y = std::numeric_limits<int>::max();
//				//		int max_x = std::numeric_limits<int>::min();
//				//		int max_y = std::numeric_limits<int>::min();
//
//				//		for (const auto& pt : edgePoints) {
//				//			min_x = std::min(min_x, pt.x);
//				//			min_y = std::min(min_y, pt.y);
//				//			max_x = std::max(max_x, pt.x);
//				//			max_y = std::max(max_y, pt.y);
//				//		}
//
//				//		// 如果要求裁剪区域的 min_x 和 min_y 为偶数，可以执行下面两句：
//				//		if (min_x % 2 == 1) --min_x;
//				//		if (min_y % 2 == 1) --min_y;
//
//				//		int width = max_x - min_x + 1;
//				//		int height = max_y - min_y + 1;
//
//				//		Rect cropRect(min_x, min_y, width, height);
//				//		cout << "Crop rect: " << cropRect << endl;
//
//				//		// 坐标平移
//				//		std::vector<cv::Point2f> shiftedPoints;
//				//		for (const auto& pt : subedgePoints) {
//				//			float new_x = pt.x - min_x;
//				//			float new_y = pt.y - min_y;
//
//				//			if (new_x >= 0 && new_y >= 0) {
//				//				shiftedPoints.emplace_back(new_x, new_y);
//				//			}
//				//		}
//				//		// 替换原来的点集合（如果需要）
//				//		subedgePoints = shiftedPoints;
//				//		string outPath = "D:/mark_window/module/algorithm/src/sub_edge_points.txt";
//				//		// 保存裁剪后的点
//				//		ofstream outFile(outPath);
//				//		if (!outFile.is_open()) {
//				//			cerr << "Failed to open output file." << endl;
//				//		}
//
//				//		for (const auto& pt : subedgePoints) {
//				//			outFile << pt.x << " " << pt.y << endl;
//				//		}
//				//		outFile.close();
//				//	}
//			
//				cv::Mat binary_img, edge1;
//				cv::threshold(mdl_pyrd, binary_img, 160, 255, cv::THRESH_BINARY);
//				cv::Canny(binary_img, edge1, 140, 200, 3, true);
//				
//				// 步骤1：腐蚀掩码，去掉边缘
//				cv::Mat msk_eroded;
//				int erosion_size = 1;  // 可以调整，越大去掉越多边缘
//				cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
//					cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
//				cv::erode(msk_pyrd, msk_eroded, element);
//
//				// 步骤2：使用腐蚀后的掩码过滤 edge1
//				cv::Mat edge_clean;
//				cv::bitwise_and(edge1, msk_eroded, edge_clean);
//				if (p == level) {
//					QuantifyEdge(mdl_pyrd, angle8, quantized_angle8, mag8, weak_thresh, false);
//					Template templ = ExtractTemplate(angle8, quantized_angle8, mag8,
//						shape_infos[s], PyramidLevel(p),
//						weak_thresh, strong_thresh,
//						features_pyrd, msk_pyrd, edge2);
//
//					templ_all_[p].push_back(templ);
//				}
//				else 
//				{
//					templ_all_[p].push_back(Template());  // normal 模板为空
//				}
//				
//
//				Mat mag180, angle180, quantized_angle180;
//				
//				//cv::Mat edge2 = CannyEdge::detectEdges(mdl_pyrd, edgePoints);
//				//cv::Mat edge_masked1;
//				//cv::bitwise_and(edge2, msk_pyrd, edge_masked1);
//				if (p == 0) {
//					QuantifyEdge(mdl_pyrd, angle180, quantized_angle180, mag180, weak_thresh, true);
//					Template templ = ExtractTemplate(angle180, quantized_angle180, mag180,
//						shape_infos[s], PyramidLevel(p),
//						weak_thresh, strong_thresh,
//						features_pyrd, msk_pyrd, edge2);
//					EdgePointWithGradient subedge;
//					cv::Mat edge3;
//					edge3 = edge2.clone();
//					int shrinkPixels = 4;
//					Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));
//
//					// 2. 腐蚀 mask，缩小白色区域
//					Mat erodedMask;
//					erode(mask, erodedMask, kernel);
//
//					// 3. 与 edge 做与操作，去掉缩小区域的边缘
//					bitwise_and(edge3, erodedMask, edge3);
//					edgePoints.clear();
//					for (int y = 0; y < edge3.rows; y++) {
//						for (int x = 0; x < edge3.cols; x++) {
//							if (edge3.at<uchar>(y, x) > 0) {
//								edgePoints.push_back(cv::Point(x, y));
//							}
//						}
//					}
//				/*	std::ofstream outFile1("D:/mark_window/module/algorithm/src/edge_points.txt");
//					for (const auto& point : edgePoints) {
//						outFile1 << point.x << " " << point.y << std::endl;
//					}
//					outFile1.close();*/
//					std::ofstream outFile1("edge_points.txt");
//					for (const auto& point : edgePoints) {
//						outFile1 << point.x << " " << point.y << std::endl;
//					}
//					outFile1.close();
//					//
//					// 4. 使用Steger算法计算亚像素点
//					//int windowSize = 3; // 可根据需要调整为3或5
//					//std::vector<cv::Point2f> subpixelPoints = facetHessianSubpixel(mdl_pyrd, edgePoints);
//					//
//					// 
//					// 
//					FacetEdgeDetector detector(5);
//					//使用facet模型进行亚像素边缘检测
//					cv::Mat edgeImage = detector.detectEdges(mdl_pyrd, 30.0f);
//					// 
//					// 
//					// 
//					// 1. 读取点
//		/*			vector<Point2f> subedgePoints;
//					ifstream inFile("D:/mark_window/module/algorithm/src/sub_edge_points.txt");
//					float x, y;
//					while (inFile >> x >> y) {
//						subedgePoints.emplace_back(x, y);
//					}
//					inFile.close();*/
//					// 1. 读取点（从当前文件夹读取sub_edge_points.txt）
//					vector<Point2f> subedgePoints;
//					ifstream inFile("sub_edge_points.txt");  // 使用相对路径
//
//					// 建议添加文件存在性检查
//					if (!inFile.is_open()) {
//						std::cerr << "无法打开sub_edge_points.txt文件，请检查是否存在" << std::endl;
//						// 可以添加错误处理逻辑，如return或其他操作
//					}
//					float x, y;
//					while (inFile >> x >> y) {
//						subedgePoints.emplace_back(x, y);
//					}
//					inFile.close();
//					// 计算裁剪区域
//					int min_x = std::numeric_limits<int>::max();
//					int min_y = std::numeric_limits<int>::max();
//					int max_x = std::numeric_limits<int>::min();
//					int max_y = std::numeric_limits<int>::min();
//
//					for (const auto& pt : edgePoints) {
//						min_x = std::min(min_x, pt.x);
//						min_y = std::min(min_y, pt.y);
//						max_x = std::max(max_x, pt.x);
//						max_y = std::max(max_y, pt.y);
//					}
//
//					// 如果要求裁剪区域的 min_x 和 min_y 为偶数，可以执行下面两句：
//					if (min_x % 2 == 1) --min_x;
//					if (min_y % 2 == 1) --min_y;
//
//					int width = max_x - min_x + 1;
//					int height = max_y - min_y + 1;
//
//					Rect cropRect(min_x, min_y, width, height);
//					cout << "Crop rect: " << cropRect << endl;
//
//					// 坐标平移
//					std::vector<cv::Point2f> shiftedPoints;
//					for (const auto& pt : subedgePoints) {
//						//float new_x = pt.x - min_x;
//						//float new_y = pt.y - min_y;
//						float new_x = pt.x ;
//						float new_y = pt.y;
//						shiftedPoints.emplace_back(new_x, new_y);
//						
//					}
//					if (s == 6)
//					{
//						std::ofstream outFile1("generate_features.txt");
//						for (const auto& pt : subedgePoints) {
//							outFile1 << pt.x << " " << pt.y << std::endl;
//						}
//						outFile1.close();
//					}
//					// 替换原来的点集合（如果需要）
//					subedgePoints = shiftedPoints;
//					for (const auto& pt : subedgePoints) {
//						subedge.pt.x = pt.x;
//						subedge.pt.y = pt.y;
//						templ.edge_features.push_back(subedge);
//					}
//					
//					templ_all_[p + 3].push_back(templ);
//				}
//				else
//				{
//					templ_all_[p + 3].push_back(Template());
//				}
//				
//				//提取原始尺寸模板下梯度的信息
//				//if (p ==0)
//				//{
//				//	cv::Mat templateGradX, templateGradY, croptemplateGradX, croptemplateGradY;
//				//	Sobel(mdl_pyrd, templateGradX, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
//				//	Sobel(mdl_pyrd, templateGradY, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
//				//	for (int y = 0; y < edge_clean.rows; y++) {
//				//		for (int x = 0; x < edge_clean.cols; x++) {
//				//			if (edge_clean.at<uchar>(y, x) > 0) {
//				//				edgePoints.push_back(cv::Point(x, y));
//				//			}
//				//		}
//				//	}
//				//	for (int i = 0; i < (int)templ.features.size(); ++i)
//				//	{
//				//		int x = templ.features[i].x;
//				//		int y = templ.features[i].y;
//				//		EdgePointWithGradient epg;
//				//		epg.pt.x = x;
//				//		epg.pt.y = y;
//				//		epg.grad_x = templateGradX.at<float>(y+templ.y, x+ templ.x);
//				//		epg.grad_y = templateGradY.at<float>(y + templ.y, x + templ.x);
//				//		templ.edge_features.push_back(epg);
//
//				//	}
//				//	//for (const auto& pt : edgePoints) {
//				//	//	int x = pt.x;
//				//	//	int y = pt.y;
//				//	//	EdgePointWithGradient epg;
//				//	//	// 坐标偏移，变成相对 roi 的坐标
//				//	//	cv::Point2i crop(templ.x, templ.y);
//				//	//	epg.pt = pt - crop;  // roi.tl() 是左上角偏移
//				//	//	//epg.pt = pt;
//				//	//	// 梯度值仍使用原图像坐标提取
//				//	//	epg.grad_x = templateGradX.at<float>(y, x);
//				//	//	epg.grad_y = templateGradY.at<float>(y, x);
//				//	//	templ.edge_features.push_back(epg);
//				//	//}
//				//}
//				//提取亚像素信息
//				//if (p == 0 && s ==5)
//				//{
//				//	/*Rect roi(templ.x, templ.y, templ.w, templ.h);
//				//	mdl_pyrd = mdl_pyrd(roi);*/
//				//	FacetEdgeDetector detector(5);
//				//	//使用facet模型进行亚像素边缘检测
//				//	cv::Mat edgeImage = detector.detectEdges(mdl_pyrd, 30.0f);
//				//	// 1. 读取点
//				//	vector<Point2f> subedgePoints;
//				//	ifstream inFile("sub_edge_points.txt");
//				//	float x, y;
//				//	while (inFile >> x >> y) {
//				//		subedgePoints.emplace_back(x, y);
//				//	}
//				//	inFile.close();
//
//				//	// 2. 定义 ROI 区域
//				//	Rect roi(templ.x, templ.y, templ.w, templ.h);
//
//
//				//	// 3. 生成新图上的坐标（平移原始点）
//				//	vector<Point2f> shiftedPoints;
//				//	for (const auto& pt : subedgePoints) {
//				//		if (roi.contains(pt)) {  // 可选：确保在裁剪范围内
//				//			Point2f shifted(pt.x - roi.x, pt.y - roi.y);
//				//			shiftedPoints.push_back(shifted);
//				//		}
//				//		else
//				//		{
//
//				//		}
//				//	}
//				//	// 4. 保存裁剪图坐标系下的点
//				//	ofstream outFile("sub_edge_points1.txt");
//				//	for (const auto& pt : shiftedPoints) {
//				//		outFile << pt.x << " " << pt.y << endl;
//				//	}
//				//	outFile.close();
//				//}
//				
//
//				/// draw
//				/*Mat draw_mask;
//				msk_pyrd.copyTo(draw_mask);
//				DrawTemplate(draw_mask, templ, Scalar(0));
//				imshow("draw_mask", draw_mask);
//				waitKey(1);*/
//			}
//			//cout << "train pyramid level " << p << " complete." << endl;
//		}
//		SaveModel();
//	}
	void KcgMatch::MakingTemplates(Mat model, AngleRange angle_range, ScaleRange scale_range,
		int num_features, int binarize_thresh, float weak_thresh, float strong_thresh, Mat mask) {
		ClearModel();
		//////////
		 // 检测矩形
		//vector<RectangleInfo> rectInfos;
		//int minArea = 8;
		//bool success = detectRectangles(model, rectInfos, minArea); // 最小面积设为200
		//////////
		std::vector<cv::Point> edgePoints;
		cv::Mat edge = CannyEdge::detectEdges(model, edgePoints, binarize_thresh);
		int edgeCount = cv::countNonZero(edge);
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
	
		PaddingModelAndMask(model, mask, scale_range.end);
		angle_range_ = angle_range;
		scale_range_ = scale_range;
		vector<ShapeInfo> shape_infos = ProduceShapeInfos(angle_range, scale_range);
		vector<Mat> l0_mdls; l0_mdls.clear();
		vector<Mat> l0_msks; l0_msks.clear();
		for (int s = 0; s < shape_infos.size(); s++) {

			l0_mdls.push_back(MdlOf(model, shape_infos[s]));
			l0_msks.push_back(MskOf(mask, shape_infos[s]));
		}
        #pragma omp parallel for num_threads(omp_get_num_procs())  // 自动获取CPU核心数
		for (int p = 0; p <= PyramidLevel_2; p++) {

			for (int s = 0; s < shape_infos.size(); s++) {

				Mat mdl_pyrd = l0_mdls[s];
				Mat msk_pyrd = l0_msks[s];
				if (p > 0) {

					Size sz = Size(l0_mdls[s].cols >> 1, l0_mdls[s].rows >> 1);
					pyrDown(l0_mdls[s], mdl_pyrd, sz);
					pyrDown(l0_msks[s], msk_pyrd, sz);
				}
				erode(msk_pyrd, msk_pyrd, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
				l0_mdls[s] = mdl_pyrd;
				l0_msks[s] = msk_pyrd;
				int features_pyrd = (int)((num_features >> p) * shape_infos[s].scale);

				Mat mag8, angle8, quantized_angle8;
				edgePoints.clear();
				cv::Mat edge2;

				cv::Canny(mdl_pyrd, edge2, 170, 200, 3, true);

				cv::Mat edge_2;
				edge_2 = edge2.clone();
				int shrinkPixels = 4;
				Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));

				// 2. 腐蚀 mask，缩小白色区域
				Mat erodedMask;
				erode(msk_pyrd, erodedMask, kernel);

				// 3. 与 edge 做与操作，去掉缩小区域的边缘
				bitwise_and(edge_2, erodedMask, edge_2);
				int whitePixelCount = cv::countNonZero(edge_2);
				if (whitePixelCount < edge2.cols * 0.1)
				{
					cv::Mat medianFilteredImage;
					cv::medianBlur(mdl_pyrd, medianFilteredImage, 3);
					cv::Mat preprocessedImage;
					cv::equalizeHist(medianFilteredImage, preprocessedImage);  // 提高对比度

					// 使用双边滤波减少噪声
					cv::Mat smoothedImage2, edges2;
					cv::bilateralFilter(preprocessedImage, smoothedImage2, 9, 75, 75);
					cv::Canny(smoothedImage2, edge2, 170, 200, 3, true);
				}
							
				if (p == level) {
					QuantifyEdge(mdl_pyrd, angle8, quantized_angle8, mag8, weak_thresh, false);
					Template templ = ExtractTemplate(angle8, quantized_angle8, mag8,
						shape_infos[s], PyramidLevel(p),
						weak_thresh, strong_thresh,
						features_pyrd, msk_pyrd, edge2);
					// 线程安全写入：每个线程操作不同的p和s索引，无竞争
                    #pragma omp critical  // 确保vector::push_back线程安全
					templ_all_[p].push_back(templ);
				}
				else
				{
                    #pragma omp critical
					templ_all_[p].push_back(Template());  // normal 模板为空
				}


				Mat mag180, angle180, quantized_angle180;

				if (p == 0) {
					//图像预处理
					vector<vector<Point>> contours;
					vector<Vec4i> hierarchy; // 存储轮廓层级关系（关键）
	// 使用RETR_CCOMP模式：提取所有轮廓，并分为两级（外部轮廓和内部轮廓）
	// 若需更详细的层级关系，可改用RETR_TREE
					findContours(edge2, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
					vector<Point> edgePoints;
					for (const auto& cnt : contours) {
						if (contourArea(cnt) > 8) {
							edgePoints.insert(edgePoints.end(), cnt.begin(), cnt.end());
						}
					}
					cv::Mat filtered_edge = cv::Mat::zeros(edge2.size(), edge2.type()); // 创建空白图像（与原边缘图同尺寸）

				  // 方法1：直接绘制所有过滤后的点（适合点集较密集的情况）
					for (const Point& pt : edgePoints) {
						if (pt.inside(Rect(0, 0, filtered_edge.cols, filtered_edge.rows))) { // 确保点在图像范围内
							filtered_edge.at<uchar>(pt) = 255; // 边缘点设为白色（255）
						}
					}
					edgePoints.clear();
					edge2 = filtered_edge.clone();
					QuantifyEdge(mdl_pyrd, angle180, quantized_angle180, mag180, weak_thresh, true);
					Template templ = ExtractTemplate(angle180, quantized_angle180, mag180,
						shape_infos[s], PyramidLevel(p),
						weak_thresh, strong_thresh,
						features_pyrd, msk_pyrd, edge2);
					EdgePointWithGradient subedge;
					cv::Mat edge3;
					edge3 = edge2.clone();
					int shrinkPixels = 4;
					Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));

					// 2. 腐蚀 mask，缩小白色区域
					Mat erodedMask;
					erode(mask, erodedMask, kernel);

					// 3. 与 edge 做与操作，去掉缩小区域的边缘
					bitwise_and(edge3, erodedMask, edge3);
					edgePoints.clear();
					for (int y = 0; y < edge3.rows; y++) {
						for (int x = 0; x < edge3.cols; x++) {
							if (edge3.at<uchar>(y, x) > 0) {
								edgePoints.push_back(cv::Point(x, y));
							}
						}
					}
					
					std::ofstream outFile1("edge_points.txt");
					for (const auto& point : edgePoints) {
						outFile1 << point.x << " " << point.y << std::endl;
					}
					outFile1.close();
					//FacetEdgeDetector detector(5);
					//使用facet模型进行亚像素边缘检测
					//cv::Mat edgeImage = detector.detectEdges(mdl_pyrd, 30.0f);

					std::vector<cv::Point2f> subedgePoints;
					SubPixelByZernike(mdl_pyrd, edgePoints, subedgePoints);

					//SubPixelBySM(mdl_pyrd, edgePoints, subedgePoints);
					// 
					//// 1. 读取点（从当前文件夹读取sub_edge_points.txt）
					//vector<Point2f> subedgePoints;
					//ifstream inFile("sub_edge_points.txt");  // 使用相对路径

					//// 建议添加文件存在性检查
					//if (!inFile.is_open()) {
					//	std::cerr << "无法打开sub_edge_points.txt文件，请检查是否存在" << std::endl;
					//	// 可以添加错误处理逻辑，如return或其他操作
					//}
					//float x, y;
					//while (inFile >> x >> y) {
					//	subedgePoints.emplace_back(x, y);
					//}
					//inFile.close();
					// 计算裁剪区域
					int min_x = std::numeric_limits<int>::max();
					int min_y = std::numeric_limits<int>::max();
					int max_x = std::numeric_limits<int>::min();
					int max_y = std::numeric_limits<int>::min();

					for (const auto& pt : edgePoints) {
						min_x = std::min(min_x, pt.x);
						min_y = std::min(min_y, pt.y);
						max_x = std::max(max_x, pt.x);
						max_y = std::max(max_y, pt.y);
					}

					// 如果要求裁剪区域的 min_x 和 min_y 为偶数，可以执行下面两句：
					if (min_x % 2 == 1) --min_x;
					if (min_y % 2 == 1) --min_y;

					int width = max_x - min_x + 1;
					int height = max_y - min_y + 1;

					Rect cropRect(min_x, min_y, width, height);

					// 坐标平移
					std::vector<cv::Point2f> shiftedPoints;
					for (const auto& pt : subedgePoints) {
						float new_x = pt.x - min_x;
						float new_y = pt.y - min_y;
					/*	float new_x = pt.x;
						float new_y = pt.y;*/
						shiftedPoints.emplace_back(new_x, new_y);

					}
					if (s == 6)
					{
						std::ofstream outFile1("generate_features.txt");
						for (const auto& pt : subedgePoints) {
							outFile1 << pt.x << " " << pt.y << std::endl;
						}
						outFile1.close();
					}
					// 替换原来的点集合（如果需要）
					subedgePoints = shiftedPoints;
					for (const auto& pt : subedgePoints) {
						subedge.pt.x = pt.x;
						subedge.pt.y = pt.y;
						templ.edge_features.push_back(subedge);
					}
                    #pragma omp critical
					templ_all_[p + 3].push_back(templ);
				}
				else
				{
                     #pragma omp critical
					templ_all_[p + 3].push_back(Template());
				}
			}
		}
		SaveModel();
	}
	void KcgMatch::MakingTemplates1(Mat model, AngleRange angle_range, ScaleRange scale_range,
		int num_features, float weak_thresh, float strong_thresh, Mat mask) {

		ClearModel();
		PaddingModelAndMask(model, mask, scale_range.end);
		angle_range_ = angle_range;
		scale_range_ = scale_range;
		vector<ShapeInfo> shape_infos = ProduceShapeInfos(angle_range, scale_range);
		vector<Mat> l0_mdls; l0_mdls.clear();
		vector<Mat> l0_msks; l0_msks.clear();
		for (int s = 0; s < shape_infos.size(); s++) {

			l0_mdls.push_back(MdlOf(model, shape_infos[s]));
			//l0_msks.push_back(MskOf(mask, shape_infos[s]));
		}
		//提取边缘点和对应的facet梯度
		FacetEdgeDetector detector(5);
		//使用facet模型进行亚像素边缘检测
		for (int p = 0; p <= PyramidLevel_0; p++) {

			for (int s = 0; s < shape_infos.size(); s++) {

				Mat mdl_pyrd = l0_mdls[s];
				//Mat msk_pyrd = l0_msks[s];
				if (p > 0) {

					Size sz = Size(l0_mdls[s].cols >> 1, l0_mdls[s].rows >> 1);
					pyrDown(l0_mdls[s], mdl_pyrd, sz);
					//pyrDown(l0_msks[s], msk_pyrd, sz);
				}
				
				cv::Mat templateGradX, templateGradY, croptemplateGradX, croptemplateGradY;
				std::vector<cv::Point> edgePoints;
				cv::Mat edge = CannyEdge::detectEdges(mdl_pyrd, edgePoints);
				//detector.calculateGradientVectors(mdl_pyrd, templateGradX, templateGradY, edgePoints);
				Sobel(mdl_pyrd, templateGradX, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		        Sobel(mdl_pyrd, templateGradY, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
				Template templ;
				templ.shape_info.angle = shape_infos[s].angle;
				templ.shape_info.scale = shape_infos[s].scale;
				templ.pyramid_level = PyramidLevel(p);
				templ.is_valid = 0;
				templ.features.clear();
				templ.edge_features.clear();
				// 存储每个边缘点和对应的梯度
				// 寻找边缘点区域的最小边界框
				int minX = edgePoints[0].x, minY = edgePoints[0].y;
				int maxX = edgePoints[0].x, maxY = edgePoints[0].y;

				for (size_t i = 1; i < edgePoints.size(); i++) {
					minX = min(minX, edgePoints[i].x);
					minY = min(minY, edgePoints[i].y);
					maxX = max(maxX, edgePoints[i].x);
					maxY = max(maxY, edgePoints[i].y);
				}
				// 裁剪出边缘点区域
				Rect roi(minX, minY, maxX - minX + 1, maxY - minY + 1);
				// 存储裁剪后的边缘点（坐标相对于裁剪区域），梯度仍从原图获取
				templ.w = roi.size().width;
				templ.h = roi.size().height;
				//cv::Mat roiGradX = templateGradX(roi);
				//cv::Mat roiGradY = templateGradY(roi);
				//cv::Mat roimdl_pyrd = mdl_pyrd(roi);
				//edgePoints.clear();
				//cv::Mat edge1 = CannyEdge::detectEdges(roimdl_pyrd, edgePoints);
				//以图像边缘点为参考点
				for (const auto& pt : edgePoints) {
					int x = pt.x;
					int y = pt.y;

					if (x >= 0 && x < templateGradX.cols && y >= 0 && y < templateGradX.rows) {
						
						if (shape_infos[s].scale == 0.5)
						{
							smallEdgePointWithGradient small_epg;
							small_epg.pt = pt - roi.tl();  // roi.tl() 是左上角偏移
							//small_epg.pt = pt;
							// 梯度值仍使用原图像坐标提取
							small_epg.grad_x = templateGradX.at<float>(y, x);
							small_epg.grad_y = templateGradY.at<float>(y, x);
						    templ.small_edge_features.push_back(small_epg);
						}
						else {
							EdgePointWithGradient epg;
							// 坐标偏移，变成相对 roi 的坐标
							epg.pt = pt - roi.tl();  // roi.tl() 是左上角偏移
							//epg.pt = pt;
							// 梯度值仍使用原图像坐标提取
							epg.grad_x = templateGradX.at<float>(y, x);
							epg.grad_y = templateGradY.at<float>(y, x);
							templ.edge_features.push_back(epg);
						}
						
					}
				}
				// -------------------------------------------------------------
				// 以图像边缘点重心为参考点
				//// // 第一步：计算边缘点重心
				//cv::Point centroid(0.f, 0.f);
				//for (const auto& pt : edgePoints) {
				//	centroid += pt;
				//}
				//centroid *= (1.0f / edgePoints.size());  // 求平均坐标
				//templ.x = centroid.x;
				//templ.y = centroid.y;
				//// 第二步：根据重心进行坐标转换和特征存储
				//for (const auto& pt : edgePoints) {
				//	int x = pt.x;
				//	int y = pt.y;

				//	if (x >= 0 && x < templateGradX.cols && y >= 0 && y < templateGradX.rows) {

				//		// 相对重心坐标
				//		cv::Point2f relative_pt = pt - centroid;

				//		if (shape_infos[s].scale == 0.5) {
				//			smallEdgePointWithGradient small_epg;
				//			small_epg.pt = relative_pt;
				//			small_epg.grad_x = templateGradX.at<float>(y, x);
				//			small_epg.grad_y = templateGradY.at<float>(y, x);
				//			templ.small_edge_features.push_back(small_epg);
				//		}
				//		else {
				//			EdgePointWithGradient epg;
				//			epg.pt = relative_pt;
				//			epg.grad_x = templateGradX.at<float>(y, x);
				//			epg.grad_y = templateGradY.at<float>(y, x);
				//			templ.edge_features.push_back(epg);
				//		}
				//	}
				//}
				//erode(msk_pyrd, msk_pyrd, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
				//l0_mdls[s] = mdl_pyrd;
				//l0_msks[s] = msk_pyrd;

				//int features_pyrd = (int)((num_features >> p) * shape_infos[s].scale);

				//Mat mag8, angle8, quantized_angle8;
				//QuantifyEdge(mdl_pyrd, angle8, quantized_angle8, mag8, weak_thresh, false);
				////Template templ = ExtractTemplate(angle8, quantized_angle8, mag8,
				////	shape_infos[s], PyramidLevel(p),
				////	weak_thresh, strong_thresh,
				////	features_pyrd, msk_pyrd);
				//templ_all_[p].push_back(templ);

				//Mat mag180, angle180, quantized_angle180;
				//QuantifyEdge(mdl_pyrd, angle180, quantized_angle180, mag180, weak_thresh, true);
				//templ = ExtractTemplate(angle180, quantized_angle180, mag180,
				//	shape_infos[s], PyramidLevel(p),
				//	weak_thresh, strong_thresh,
				//	features_pyrd, msk_pyrd);
				//templ_all_[p + 8].push_back(templ);
				templ_all_[p + s].push_back(templ);

				/// draw
				/*Mat draw_mask;
				msk_pyrd.copyTo(draw_mask);
				DrawTemplate(draw_mask, templ, Scalar(0));
				imshow("draw_mask", draw_mask);
				waitKey(1);*/
			}
			//cout << "train pyramid level " << p << " complete." << endl;
		}
		SaveModel1();
	}
	void KcgMatch::MakingTemplates_fast(Mat model, AngleRange angle_range, ScaleRange scale_range,
		int num_features, float weak_thresh, float strong_thresh, Mat mask) {

		ClearModel();
		std::vector<cv::Point> edgePoints;
		cv::Mat edge = CannyEdge::detectEdges(model, edgePoints);
		// 寻找边缘点区域的最小边界框
		int minX, minY, maxX, maxY;
		if (edgePoints.size() > 0)
		{
			minX = edgePoints[0].x, minY = edgePoints[0].y;
			maxX = edgePoints[0].x, maxY = edgePoints[0].y;
			for (size_t i = 1; i < edgePoints.size(); i++) {
				minX = min(minX, edgePoints[i].x);
				minY = min(minY, edgePoints[i].y);
				maxX = max(maxX, edgePoints[i].x);
				maxY = max(maxY, edgePoints[i].y);
			}
		}
		
		if (minX % 2 == 1) --minX;
		if (minY % 2 == 1) --minY;

		cv::Rect roi_save(minX, minY, maxX - minX + 2, maxY - minY + 2);
		std::string outPath = "D:/mark_window/module/algorithm/src/generate_features.txt";
		std::ofstream outFile(outPath);
		if (!outFile.is_open()) {
			return;
		}
		else {
			for (int i = 0; i < edgePoints.size(); ++i) {
				edgePoints[i].x -= roi_save.x;
				edgePoints[i].y -= roi_save.y;
				outFile << edgePoints[i].x << " " << edgePoints[i].y << endl;
			}
			outFile.close();
		}
		

		// 裁剪出边缘点区域
		//Rect roi(minX, minY, maxX - minX + 2, maxY - minY + 2);
		Rect roi(minX - 60, minY - 60, maxX - minX + 120, maxY - minY + 120);
		model = model(roi);
		PaddingModelAndMask(model, mask, scale_range.end);
		angle_range_ = angle_range;
		scale_range_ = scale_range;
		vector<ShapeInfo> shape_infos = ProduceShapeInfos(angle_range, scale_range);
		vector<Mat> l0_mdls; l0_mdls.clear();
		vector<Mat> l0_msks; l0_msks.clear();
		
		for (int s = 0; s < shape_infos.size(); s++) {

			l0_mdls.push_back(MdlOf(model, shape_infos[s]));
			l0_msks.push_back(MskOf(mask, shape_infos[s]));
		}
		//cv::Mat edge_final = CannyEdge::detectEdges(l0_mdls[10], edgePoints);
		for (int p = 0; p <= PyramidLevel_2; p++) {
			Mat mdl_pyrd = l0_mdls[10];
			Mat msk_pyrd = l0_msks[10];
			if (p > 0) {

				Size sz = Size(l0_mdls[10].cols >> 1, l0_mdls[10].rows >> 1);
				pyrDown(l0_mdls[10], mdl_pyrd, sz);
				pyrDown(l0_msks[10], msk_pyrd, sz);
				//pyrDown(edge_final, edge_final);
			}

			erode(msk_pyrd, msk_pyrd, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
			l0_mdls[10] = mdl_pyrd;
			l0_msks[10] = msk_pyrd;

			int features_pyrd = (int)((num_features >> p) * shape_infos[0].scale);

			Mat mag8, angle8, quantized_angle8;
			edgePoints.clear();
			cv::Mat edge_final = CannyEdge::detectEdges(mdl_pyrd, edgePoints);
			cv::Mat binary_img, edge1;
			cv::threshold(mdl_pyrd, binary_img, 160, 255, cv::THRESH_BINARY);
			cv::Canny(binary_img, edge1, 140, 200, 3, true);
			QuantifyEdge(mdl_pyrd, angle8, quantized_angle8, mag8, weak_thresh, false);
			// 步骤1：腐蚀掩码，去掉边缘
			cv::Mat msk_eroded;
			int erosion_size = 1;  // 可以调整，越大去掉越多边缘
			cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
				cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
			cv::erode(msk_pyrd, msk_eroded, element);

			// 步骤2：使用腐蚀后的掩码过滤 edge1
			cv::Mat edge_clean;
			cv::bitwise_and(edge1, msk_eroded, edge_clean);

			Template templ = ExtractTemplate(angle8, quantized_angle8, mag8,
				shape_infos[10], PyramidLevel(p),
				weak_thresh, strong_thresh,
				features_pyrd, msk_pyrd, edge_final);

			templ_all_[p].push_back(templ);

			Mat mag180, angle180, quantized_angle180;
			QuantifyEdge(mdl_pyrd, angle180, quantized_angle180, mag180, weak_thresh, true);
			//cv::Mat edge2 = CannyEdge::detectEdges(mdl_pyrd, edgePoints);
			//cv::Mat edge_masked1;
			//cv::bitwise_and(edge2, msk_pyrd, edge_masked1);
			templ = ExtractTemplate(angle180, quantized_angle180, mag180,
				shape_infos[10], PyramidLevel(p),
				weak_thresh, strong_thresh,
				features_pyrd, msk_pyrd, edge_final);
			templ_all_[p + 3].push_back(templ);
			for (int s = 0; s < shape_infos.size(); s++) {
				float theta = shape_infos[s].angle;
				//旋转0度特征点
				addTemplate_rotate(p, theta);
			}
			templ_all_[p].erase(templ_all_[p ].begin());
			templ_all_[p + 3].erase(templ_all_[p + 3].begin());
			//cout << "train pyramid level " << p << " complete." << endl;
		}
		SaveModel();
	}
	vector<Match> KcgMatch::Matching(Mat source, float score_thresh, float overlap,
		float mag_thresh, float greediness, PyramidLevel pyrd_level, int T, int top_k,
		MatchingStrategy strategy, const Mat mask) {
		
		InitMatchParameter(score_thresh, overlap, mag_thresh, greediness, T, top_k, strategy);
		GetAllPyramidLevelValidSource(source, pyrd_level);

		vector<Match> matches;
		matches = MatchingPyrd8(sources_[pyrd_level], pyrd_level, region8_idxes_);
		matches = GetTopKMatches(matches);
		////添加压缩2层时匹配
		//for (int i = 0; i < matches.size(); i++) {
		//	if (matches[i].template_id > 0)
		//	{
		//		//region8_idxes_.push_back(matches[i].template_id - 1);
		//		region8_idxes_.push_back(matches[i].template_id);
		//	}
		//	//matches = MatchingPyrd8(sources_[pyrd_level - 2], PyramidLevel( pyrd_level-2), region8_idxes_);
		//	auto tmp_matches = MatchingPyrd180(sources_[pyrd_level], PyramidLevel_2, region8_idxes_);
		//	matches = GetTopKMatches(matches);
		//}
		
		//matches = ReconfirmMatches(matches, pyrd_level);
		//matches = GetTopKMatches(matches);
		int64 start = cv::getTickCount();  // 计时器
		matches = MatchingFinal(matches, pyrd_level);
		matches = GetTopKMatches(matches);
		int64 end = cv::getTickCount();
		double time = (end - start) / static_cast<double>(cv::getTickFrequency()) * 1000;
		// 执行精确偏移求解
		if (matches.size() > 0)
		{
			auto match = matches[0];
			//// 在使用template_id的地方添加角度映射转换
			//int originalTemplateId = match.template_id;
			//int reversedTemplateId = 20 - originalTemplateId; // 反转映射：20-15=5，20-5=15
			//auto templ = templ_all_[3][reversedTemplateId];
			auto templ = templ_all_[3][match.template_id];
			//auto templ = templ_all_[3][10];
			std::vector<cv::Point2f> points;
			for (int i = 0; i < (int)templ.features.size(); i++) {

				auto feature = templ.features[i];
				points.push_back(cv::Point2f(feature.x, feature.y));
			}
			vector<Point2f> rotated_points;
			for (int i = 0; i < (int)templ.edge_features.size(); i++) {

				auto edge_features = templ.edge_features[i];
				rotated_points.push_back(cv::Point2f(edge_features.pt.x, edge_features.pt.y));
			}
			//方法一：雅可比行列式，距离对齐
			cv::Mat float_img;
			//rotated_points = ReadAndRotateEdgePoints("D:/mark_window/module/algorithm/src/sub_edge_points.txt", Size(302, 302), 30.0f, 1.0f);
			cv::Mat edge_image = cv::Mat::zeros(sources_[PyramidLevel_0].size(), CV_8UC1);
			cv::Canny(sources_[PyramidLevel_0], edge_image, 150, 250, 3, false);
			cv::Mat image_edge_fanse, image_edge_distance;
			cv::bitwise_not(edge_image, image_edge_fanse, cv::noArray());
			//进行距离变换
			cv::distanceTransform(image_edge_fanse, image_edge_distance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
			exp(-0.2 * image_edge_distance, image_edge_distance);
			image_edge_distance = 1 - image_edge_distance;
			r1 = 0.0;   // 旋转角度
			x1 = match.x;   // 平移 x
			y1 = match.y;   // 平移 y
		   // 模板中心在模板坐标中的位置
			float cx = templ.w / 2.0f;
			float cy = templ.h / 2.0f;
			auto feature = templ.features;
			//int result = precise_offset_solver::findPreciseOffset(image_edge_distance, points, r1, x1, y1);
			//方法二：使用icp刚性对齐
			FacetEdgeDetector detector(5);
			//使用facet模型进行亚像素边缘检测
			//cv::Mat edgeImage = detector.detectEdges(sources_[PyramidLevel_0], 30.0f);
			//vector<Point2f> modelPoints = ReadAndRotateEdgePoints("sub_edge_points.txt", Size(302, 302), 30.0f, 1.0f);
			//int status = precise_offset_solver::findRigidTransform2D(rotated_points, modelPoints, r, x, y,50, 1e-5);
			//方法三：点到面icp匹配
			sm::imgproc::Scene_edge scene;
			vector<sm::imgproc::Vec2f> pcd_buffer, normal_buffer;
			scene.init_Scene_edge(sources_[PyramidLevel_0], pcd_buffer, normal_buffer);
			std::vector<sm::imgproc::Vec2f> model_pcd(rotated_points.size());
			for (int i = 0; i < rotated_points.size(); i++)
			{
				auto& feat = rotated_points;
				model_pcd[i] = { float(rotated_points[i].x + match.x ), float(rotated_points[i].y + match.y) };
			}
			sm::imgproc::icp::RegistrationResult Reresult = sm::imgproc::icp::ICP2D_Point2Plane(model_pcd, scene);
			float x_refine = Reresult.transformation_[0][0] * match.x  + Reresult.transformation_[0][1] * match.y  + Reresult.transformation_[0][2];
			float y_refine = Reresult.transformation_[1][0] * match.x + Reresult.transformation_[1][1] * match.y + Reresult.transformation_[1][2];
			float cos_r = Reresult.transformation_[0][0];
			float sin_r = Reresult.transformation_[1][0];
			float r_refine = std::atan2(sin_r, cos_r); // r 是旋转角，单位是弧度
			float r_deg = r_refine * 180.0f / M_PI; // 弧度转角度
			x = x_refine;
			y = y_refine;
			r = r_deg;
			if (matches.size() > 0)
			{
				matches[0].x = x;
				matches[0].y = y;
				matches[0].r = r;

			}
		}
		
		return matches;
	}
	vector<Match> KcgMatch::Matching1(Mat source, float score_thresh, float overlap,
		float mag_thresh, float greediness, PyramidLevel pyrd_level, int T, int top_k,
		MatchingStrategy strategy, const Mat mask) {
		vector<Match> matches;
		// 图像预处理
		FacetEdgeDetector detector(5);
		cv::Mat grad_x, grad_y, source_pyrd, grad_x_pyrd, grad_y_pyrd;
		pyrDown(source, source_pyrd, Size(source.cols >> 1, source.rows >> 1));
		Sobel(source, grad_x, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		Sobel(source, grad_y, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		Sobel(source_pyrd, grad_x_pyrd, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		Sobel(source_pyrd, grad_y_pyrd, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		/*detector.calculateGradientVectors(source, grad_x, grad_y);
		detector.calculateGradientVectors(source_pyrd, grad_x_pyrd, grad_y_pyrd);*/
		// 归一化梯度（原图）
		Mat norm_grad_x = grad_x.clone(), norm_grad_y = grad_y.clone();
#pragma omp parallel for collapse(2)
		for (int y = 0; y < grad_y.rows; ++y) {
			for (int x = 0; x < grad_x.cols; ++x) {
				float gx = grad_x.at<float>(y, x);
				float gy = grad_y.at<float>(y, x);
				float mag = sqrt(gx * gx + gy * gy);
				/*if (mag > 1e-5f) {*/
					norm_grad_x.at<float>(y, x) = gx / mag;
					norm_grad_y.at<float>(y, x) = gy / mag;
				/*}
				else {
					norm_grad_x.at<float>(y, x) = 0.0f;
					norm_grad_y.at<float>(y, x) = 0.0f;
				}*/
			}
		}

		// 归一化梯度（金字塔图）
		Mat norm_grad_x_pyrd = grad_x_pyrd.clone(), norm_grad_y_pyrd = grad_y_pyrd.clone();
#pragma omp parallel for collapse(2)
		for (int y = 0; y < grad_x_pyrd.rows; ++y) {
			for (int x = 0; x < grad_x_pyrd.cols; ++x) {
				float gx = grad_x_pyrd.at<float>(y, x);
				float gy = grad_y_pyrd.at<float>(y, x);
				float mag = sqrt(gx * gx + gy * gy);
				//if (mag > 1e-5f) {
					norm_grad_x_pyrd.at<float>(y, x) = gx / mag;
					norm_grad_y_pyrd.at<float>(y, x) = gy / mag;
				/*}
				else {
					norm_grad_x_pyrd.at<float>(y, x) = 0.0f;
					norm_grad_y_pyrd.at<float>(y, x) = 0.0f;
				}*/
			}
		}

		// ========== 阶段 1：金字塔图粗匹配 ==========
		struct Candidate { Point pt; int angleIdx; double score; };
		vector<Candidate> coarse_candidates;
		double small_theastep = 1;  // 模板获取间隔值
		double bestscore = -1;
		int step = 4;
#pragma omp parallel for

				for (int angleIdx = 0; angleIdx < 21; angleIdx+= small_theastep) { // 前21个为金字塔模板
					for (const auto& templ : templ_all_[angleIdx]) {
						if (templ.small_edge_features.empty()) continue;

						const int tplWidth = templ.w;
						const int tplHeight = templ.h;
						const int rangeX = norm_grad_x_pyrd.cols - tplWidth;
						const int rangeY = norm_grad_y_pyrd.rows - tplHeight;
						for (int y = 0; y < rangeY; y += step) {
							for (int x = 0; x < rangeX; x += step) {
								float totalSim = 0.0f;
								int validCount = 0;
								for (const auto& ep : templ.small_edge_features) {
									int px = x + ep.pt.x;
									int py = y + ep.pt.y;
									if (px < 0 || px >= norm_grad_x_pyrd.cols || py < 0 || py >= norm_grad_y_pyrd.rows)
										continue;

									float gnx = norm_grad_x_pyrd.at<float>(py, px);
									float gny = norm_grad_y_pyrd.at<float>(py, px);
									float magTemplate = std::sqrt(ep.grad_x * ep.grad_x + ep.grad_y * ep.grad_y);
									//if (magTemplate > 1e-9) {  // 只考虑非零梯度的像素
										float gtx = ep.grad_x / magTemplate;
										float gty = ep.grad_y / magTemplate;
										totalSim += gnx * gtx + gny * gty;
										validCount++;
									//}
								}
								if (validCount > 0) {
									float avgSim = totalSim / validCount;
									if (avgSim > bestscore) { // 可调节粗匹配阈值
										 bestscore=avgSim;
										coarse_candidates.push_back({ cv::Point(x * 2, y * 2), angleIdx,bestscore });
									}
								}
					}
				}
			}
		}
//#pragma omp parallel for
//		for (int angleIdx = 0; angleIdx < 21; angleIdx += small_theastep) {
//			for (const auto& templ : templ_all_[angleIdx]) {
//				if (templ.small_edge_features.empty()) continue;
//
//				// 获取模板重心（必须预先在模板构建时存入）
//				cv::Point2f centroid; 
//				centroid.x = templ.x;
//				centroid.y = templ.y;
//				const int tplWidth = templ.w;
//				const int tplHeight = templ.h;
//				const int rangeX = norm_grad_x_pyrd.cols - tplWidth;
//				const int rangeY = norm_grad_y_pyrd.rows - tplHeight;
//
//				for (int y = 0; y < rangeY; y += step) {
//					for (int x = 0; x < rangeX; x += step) {
//
//						float totalSim = 0.0f;
//						int validCount = 0;
//
//						for (const auto& ep : templ.small_edge_features) {
//							// 使用重心为参考点的匹配坐标
//							int px = static_cast<int>(x + ep.pt.x + 0.5f - centroid.x);
//							int py = static_cast<int>(y + ep.pt.y + 0.5f - centroid.y);
//
//							if (px < 0 || px >= norm_grad_x_pyrd.cols || py < 0 || py >= norm_grad_y_pyrd.rows)
//								continue;
//
//							float gnx = norm_grad_x_pyrd.at<float>(py, px);
//							float gny = norm_grad_y_pyrd.at<float>(py, px);
//							float magTemplate = std::sqrt(ep.grad_x * ep.grad_x + ep.grad_y * ep.grad_y);
//
//							float gtx = ep.grad_x / magTemplate;
//							float gty = ep.grad_y / magTemplate;
//							totalSim += gnx * gtx + gny * gty;
//							validCount++;
//						}
//
//						if (validCount > 0) {
//							float avgSim = totalSim / validCount;
//							if (avgSim > bestscore) {
//#pragma omp critical
//								{
//									if (avgSim > bestscore) {
//										bestscore = avgSim;
//										coarse_candidates.push_back({ cv::Point(x * 2, y * 2), angleIdx, bestscore });
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}

		// ========== 阶段 2：原图精匹配 ==========
		MatchResult bestResult;
		bestResult.score = -1.0f;
		step = 1;
		int angleid = 0;
#pragma omp parallel for
		if (!coarse_candidates.empty()) {
			auto& candidate = coarse_candidates.back(); // 只取最后一个候选点
			Point pt = candidate.pt;
			int coarseAngleIdx = candidate.angleIdx;

			// 原图的精匹配角度索引范围：[coarse + 21 - 3, coarse + 21 + 3]
			int baseAngleIdx = coarseAngleIdx + 21;

			for (int offset =0; offset <= 0; ++offset) {
				int angleIdx = baseAngleIdx + offset;
				if (angleIdx < 21 || angleIdx >= 42) continue; // 精匹配角度必须在[21, 41]

				for (const auto& templ : templ_all_[angleIdx]) {
					if (templ.edge_features.empty()) continue;

					
					const int tplWidth = templ.w+1;
					const int tplHeight = templ.h+1;
					const int rangeX = norm_grad_x.cols - tplWidth;
					const int rangeY = norm_grad_y.rows - tplHeight;
					for (int y = 0; y < rangeY; y += step) {
						for (int x = 0; x < rangeX; x += step) {
							float totalSim = 0.0f;
							int validCount = 0;
							for (const auto& ep : templ.edge_features) {
								int px = x + ep.pt.x;
								int py = y + ep.pt.y;

								if (px < 0 || px >= norm_grad_x.cols || py < 0 || py >= norm_grad_y.rows)
									continue;

								float gnx = norm_grad_x.at<float>(py, px);
								float gny = norm_grad_y.at<float>(py, px);
								if (std::isnan(gnx) || std::isnan(gny)) continue;

								float magTemplate = std::sqrt(ep.grad_x * ep.grad_x + ep.grad_y * ep.grad_y);
								//if (magTemplate > 1e-7) {  // 只考虑非零梯度的像素
									float gtx = ep.grad_x / magTemplate;
									float gty = ep.grad_y / magTemplate;
									if (std::isnan(ep.grad_x) || std::isnan(ep.grad_y)) continue;

									totalSim += gnx * gtx + gny * gty;
									validCount++;
								//}
							}

							if (validCount > 0) {
								double avgSim = totalSim / validCount;
#pragma omp critical
								{
									if (avgSim > bestResult.score) {
										bestResult.score = avgSim;
										pt.x = x;
										pt.y = y;
										bestResult.match_pos = pt;
										bestResult.match_angle = templ.shape_info.angle;
										angleid = angleIdx;
									}
								}
							}
						}
					}
				}
			}
		}

//#pragma omp parallel for collapse(2)
//		for (int angleIdx = 0; angleIdx < PyramidLevel_TabooUse; ++angleIdx) {
//			for (size_t t = 0; t < templ_all_[angleIdx].size(); ++t) {
//				const auto& templ = templ_all_[angleIdx][t];
//				if (templ.edge_features.empty()) continue;
//
//				const int tplWidth = templ.w;
//				const int tplHeight = templ.h;
//				const int rangeX = searchRangeX - tplWidth;
//				const int rangeY = searchRangeY - tplHeight;
//
//				for (int y = 0; y < rangeY; ++y) {
//					for (int x = 0; x < rangeX; ++x) {
//						float totalSim = 0.0f;
//						int validCount = 0;
//
//						for (const auto& ep : templ.edge_features) {
//							int px = x + ep.pt.x;
//							int py = y + ep.pt.y;
//
//							if (px < 0 || px >= grad_x.cols || py < 0 || py >= grad_x.rows)
//								continue;
//
//							float gx = grad_x.at<float>(py, px);
//							float gy = grad_y.at<float>(py, px);
//							float magTarget = std::sqrt(gx * gx + gy * gy);
//							float magTemplate = std::sqrt(ep.grad_x * ep.grad_x + ep.grad_y * ep.grad_y);
//
//							if (magTarget < 1e-5f || magTemplate < 1e-5f)
//								continue;
//
//							float gnx = gx / magTarget;
//							float gny = gy / magTarget;
//							float gtx = ep.grad_x / magTemplate;
//							float gty = ep.grad_y / magTemplate;
//
//							float dot = gnx * gtx + gny * gty;
//							totalSim += dot;
//							validCount++;
//						}
//
//						if (validCount > 0) {
//							float avgSim = totalSim / validCount;
//
//#pragma omp critical
//							{
//								if (avgSim > bestResult.score) {
//									bestResult.score = avgSim;
//									bestResult.match_pos = cv::Point(x, y);
//									bestResult.match_angle = templ.shape_info.angle;
//								}
//							}
//						}
//					}
//				}
//			}
//		}


				//if (match_count > 30) {
				//	float avg_score = total_score / match_count;
				//	// 保存角度、score、位置信息等
				//}


		cv::Mat img_with_edges;
		cv::cvtColor(source, img_with_edges, cv::COLOR_GRAY2BGR);

		const auto& templates = templ_all_[angleid];

		for (const auto& templ : templates) {
			if (templ.shape_info.angle != bestResult.match_angle)
				continue;

			for (const auto& ep : templ.edge_features) {
				cv::Point draw_pt = bestResult.match_pos + ep.pt;

				if (draw_pt.inside(cv::Rect(0, 0, img_with_edges.cols, img_with_edges.rows))) {
					cv::circle(img_with_edges, draw_pt, 0.5, cv::Scalar(0, 255, 0), -1); // 绿色点
				}
			}

			break; // 只绘制一个模板
		}

		return matches;
	}
	void KcgMatch::DrawMatches(Mat& image, vector<Match> matches, Scalar color) {

		//#pragma omp parallel for
		//vector<Point2f> rotated_points = ReadAndRotateEdgePoints("sub_edge_points1.txt", Size(302, 302), 30.0f, 1.0f);
		for (int i = 0; i < matches.size(); i++) {

			auto match = matches[i];
			//int originalTemplateId = match.template_id;
			//int reversedTemplateId = 20 - originalTemplateId; // 反转映射：20-15=5，20-5=15
			//auto templ = templ_all_[3][reversedTemplateId];
			auto templ = templ_all_[3][match.template_id];
			//auto templ = templ_all_[3][10];
			int w = match.x + templ.w;
			int h = match.y + templ.h;
			//for (int i = 0; i < (int)templ.features.size(); i++) {

			//	auto feature = templ.features[i];
			//	//circle(image, cv::Point(match.x + feature.x, match.y + feature.y), 1, color, 1);
			//	line(image,
			//		Point(match.x+ feature.x, match.y+feature.y),
			//		Point(match.x+ feature.x, match.y+feature.y),
			//		color, 1);
			//}
			double anglerad = r * CV_PI / 180.0; // 角度转弧度
			double cos_r = std::cos(anglerad);
			double sin_r = std::sin(anglerad);
			//for (int i = 0; i < rotated_points.size(); i++)
			//{
			//	double rotated_x = rotated_points[i].x * cos_r - rotated_points[i].y * sin_r + x;
			//	double rotated_y = rotated_points[i].x * sin_r + rotated_points[i].y * cos_r + y;
			//	line(image,
			//		Point(rotated_x, rotated_y),
			//		Point(rotated_x, rotated_y),
			//		color, 1);
			//}
			//std::string outPath = "D:/mark_window/module/algorithm/src/template_features.txt";
			//std::ofstream outFile(outPath);
			//if (!outFile.is_open()) {
			//	return ;
			//}
			//else {
			//	for (size_t i = 0; i < templ.edge_features.size(); ++i) {
			//		auto& f = templ.edge_features[i];
			//		outFile << f.pt.x << " " << f.pt.y << endl;
			//	}
			//	outFile.close();
			//}
			// 改为当前目录下的template_features.txt
			std::string outPath = "template_features.txt";
			std::ofstream outFile(outPath);
			if (!outFile.is_open()) {
				std::cerr << "无法创建或打开文件: " << outPath << std::endl;
				return;
			}
			else {
				for (size_t i = 0; i < templ.edge_features.size(); ++i) {
					auto& f = templ.edge_features[i];
					outFile << f.pt.x << " " << f.pt.y << std::endl;  // 建议使用std::endl明确命名空间
				}
				outFile.close();
			}
			for (int i = 0; i < (int)templ.features.size(); i++) {
				auto feature = templ.features[i];

				// 旋转后平移
				double rotated_x = feature.x * cos_r - feature.y * sin_r + match.x;
				double rotated_y = feature.x * sin_r + feature.y * cos_r + match.y;

				// 绘制点
				//cv::circle(image, cv::Point(cvRound(rotated_x), cvRound(rotated_y)), 1, color, 1);
				line(image,
					Point(rotated_x, rotated_y),
					Point(rotated_x, rotated_y),
					color, 1);
			}
			// 将图像保存到当前文件夹
			std::string savePath = "result_image.jpg";  // 相对路径，保存到当前工作目录
			bool saveSuccess = cv::imwrite(savePath, image);
			//cv::rectangle(image, { match.x, match.y }, { w, h }, color, 1);
			float angle_rad = r * CV_PI / 180.0f; // 弧度
			float cosA = std::cos(angle_rad);
			float sinA = std::sin(angle_rad);
			double r_ = -5 -r;
			// 模板中心在模板坐标中的位置
			float cx = templ.w / 2.0f;
			float cy = templ.h / 2.0f;
			matches[i].r = templ.shape_info.angle - r;
			// 旋转后加到左上角坐标
			float center_x = match.x + (cosA * cx - sinA * cy);
			float center_y = match.y + (sinA * cx + cosA * cy);
			float angle_rad1 = r1 * CV_PI / 180.0f; // 弧度
			double r_1 = -5 - r1;
			float cosA1 = std::cos(angle_rad1);
			float sinA1 = std::sin(angle_rad1);
			float center_x1 = x1 + (cosA1 * cx - sinA1 * cy);
			float center_y1 = y1 + (sinA1 * cx + cosA1 * cy);
			char info[128];
			/*sprintf(info,
				"%.2f%% [%.2f, %.2f]",
				match.similarity * 100,
				templ.shape_info.angle,
				templ.shape_info.scale);*/

		}
	}
	cv::Mat KcgMatch::DrawMatches1(cv::Mat& image, std::vector<Match>& matches, cv::Scalar color) {
		// 保留原有的绘制逻辑
		for (int i = 0; i < matches.size(); i++) {
			auto match = matches[i];
			auto templ = templ_all_[3][match.template_id];
			int w = match.x + templ.w;
			int h = match.y + templ.h;

			double anglerad = r * CV_PI / 180.0; // 角度转弧度
			double cos_r = std::cos(anglerad);
			double sin_r = std::sin(anglerad);
			float angle_rad = r * CV_PI / 180.0f;
			float cosA = std::cos(angle_rad);
			float sinA = std::sin(angle_rad);
			double r_ = -5 - r;
			float cx = templ.w / 2.0f;
			float cy = templ.h / 2.0f;
			// 保存特征点到当前目录
			std::string outPath = "template_features.txt";
			std::ofstream outFile(outPath);
			if (!outFile.is_open()) {
				std::cerr << "无法创建或打开文件: " << outPath << std::endl;
				return image; // 返回当前图像
			}
			else {
				for (size_t i = 0; i < templ.edge_features.size(); ++i) {
					auto& f = templ.edge_features[i];
					outFile << f.pt.x - (cosA * cx - sinA * cy)<< " " << f.pt.y -(sinA * cx + cosA * cy) << std::endl;
				}
				outFile.close();
			}

			// 绘制旋转后的特征点
			for (int i = 0; i < (int)templ.features.size(); i++) {
				auto feature = templ.features[i];
				// 旋转后平移
				double rotated_x = feature.x * cos_r - feature.y * sin_r + match.x;
				double rotated_y = feature.x * sin_r + feature.y * cos_r + match.y;
				// 绘制点
				cv::line(image,
					cv::Point(rotated_x, rotated_y),
					cv::Point(rotated_x, rotated_y),
					color, 1);
			}
			// 其他原有计算逻辑

			matches[i].r = templ.shape_info.angle - r;
			float center_x = match.x + (cosA * cx - sinA * cy);
			float center_y = match.y + (sinA * cx + cosA * cy);
			matches[i].x = center_x;
			matches[i].y = center_y;
			float angle_rad1 = r1 * CV_PI / 180.0f;
			double r_1 = -5 - r1;
			float cosA1 = std::cos(angle_rad1);
			float sinA1 = std::sin(angle_rad1);
			float center_x1 = x1 + (cosA1 * cx - sinA1 * cy);
			float center_y1 = y1 + (sinA1 * cx + cosA1 * cy);
			char info[128];
		}

		// 返回处理后的图像
		return image;
	}

	vector<Point2f>KcgMatch::ReadAndRotateEdgePoints(const string& filename, Size imageSize, float angle_deg, float scale)
	{
		vector<Point2f> rotated_points;
		ifstream inFile(filename);
		if (!inFile.is_open())
		{
			cerr << "Failed to open file: " << filename << endl;
			return rotated_points;
		}

		// 准备旋转矩阵
		Point2f center(imageSize.width / 2.0f, imageSize.height / 2.0f);
		Mat rot_mat = getRotationMatrix2D(center, angle_deg, scale);

		float x, y;
		while (inFile >> x >> y)
		{
			Mat pt = (Mat_<double>(3, 1) << x, y, 1.0);
			Mat rotated_pt = rot_mat * pt;

			//rotated_points.emplace_back(rotated_pt.at<double>(0, 0), rotated_pt.at<double>(1, 0));
			rotated_points.emplace_back(x, y);
		}

		return rotated_points;
	}
	//void KcgMatch::saveMatchesToXml(const std::string& filePath, const std::vector<Match>& matches) {
	//	// 创建一个 OpenCV FileStorage 对象以便写入 XML
	//	cv::FileStorage fs(filePath, cv::FileStorage::WRITE);

	//	// 保存 matches 数组
	//	fs << "matches" << "[";
	//	for (const auto& match : matches) {
	//		auto templ = templ_all_[3][match.template_id];
	//		int w = match.x + templ.w;
	//		int h = match.y + templ.h;

	//		// 保存 match 信息
	//		fs << "{";
	//		fs << "match_x" << match.x;
	//		fs << "match_y" << match.y;
	//		fs << "angle" << templ.shape_info.angle;
	//		fs << "w" << w;
	//		fs << "h" << h;

	//		// 保存 features 数组
	//		fs << "features" << "[";
	//		for (const auto& feature : templ.features) {
	//			fs << "{";
	//			fs << "x" << feature.x;
	//			fs << "y" << feature.y;
	//			fs << "}";  // 结束 feature 块
	//		}
	//		fs << "]"; // 结束 features 数组

	//		fs << "}"; // 结束 match 块
	//	}
	//	fs << "]"; // 结束 matches 数组

	//	fs.release(); // 关闭文件
	//}

	void  KcgMatch::saveMatchesToXml(const std::string& filePath, const std::vector<Match>& matches)
	{
		std::ofstream file(filePath);
		if (!file.is_open()) {
			std::cerr << "无法打开文件进行写入: " << filePath << std::endl;
			return;
		}

		file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
		file << "<matches>\n";

		for (const auto& match : matches) {
			const auto& templ = templ_all_[3][match.template_id];

			int w = match.x + templ.w;
			int h = match.y + templ.h;

			file << "  <match>\n";
			file << "    <match_x>" << x << "</match_x>\n";
			file << "    <match_y>" << y << "</match_y>\n";
			file << "    <angle>" << r << "</angle>\n";
			file << "    <w>" << w << "</w>\n";
			file << "    <h>" << h << "</h>\n";

			file << "    <features>\n";
			//for (const auto& feature : templ.features) {
			//	file << "      <feature x=\"" << feature.x << "\" y=\"" << feature.y << "\" />\n";
			//}
			vector<Point2f> rotated_points = ReadAndRotateEdgePoints("sub_edge_points.txt", Size(302, 302), 30.0f, 1.0f);
			for (const auto& feature : rotated_points) {
				file << "      <feature x=\"" << feature.x << "\" y=\"" << feature.y << "\" />\n";
			}
			file << "    </features>\n";

			file << "  </match>\n";
		}

		file << "</matches>\n";
		file.close();
	}

	void KcgMatch::PaddingModelAndMask(Mat& model, Mat& mask, float max_scale) {

		CV_Assert(!model.empty() && "model is empty.");
		if (mask.empty())
			mask = Mat(model.size(), CV_8UC1, { 255 });
		else
			CV_Assert(model.size() == mask.size());
		int min_side_length = std::min(model.rows, model.cols);
		int diagonal_line_length =
			(int)ceil(std::sqrt(model.rows * model.rows + model.cols * model.cols) * max_scale);
		int padding = ((diagonal_line_length - min_side_length) >> 1) + 16;
		int double_padding = (padding << 1);
		/*int padding = 0;
		int double_padding = 0;*/
		Mat model_padded = Mat(model.rows + double_padding, model.cols + double_padding, model.type(), Scalar::all(0));
		model.copyTo(model_padded(Rect(padding, padding, model.cols, model.rows)));
		Mat mask_padded = Mat(mask.rows + double_padding, mask.cols + double_padding, mask.type(), Scalar::all(0));
		mask.copyTo(mask_padded(Rect(padding, padding, mask.cols, mask.rows)));
		model = model_padded;
		mask = mask_padded;
	}

	vector<ShapeInfo> KcgMatch::ProduceShapeInfos(AngleRange angle_range, ScaleRange scale_range) {

		assert(scale_range.begin > KCG_EPS && scale_range.end > KCG_EPS);
		assert(angle_range.end >= angle_range.begin);
		assert(scale_range.end >= scale_range.begin);
		assert(angle_range.step > KCG_EPS);
		//assert(scale_range.step > KCG_EPS);
		vector<ShapeInfo> shape_infos;
		shape_infos.clear();
		//for (float scale = scale_range.begin; scale <= scale_range.end + KCG_EPS; scale += scale_range.step) {
		for (float scale = scale_range.begin; scale <= scale_range.end + KCG_EPS; scale += 1) {
			for (float angle = angle_range.begin; angle <= angle_range.end + KCG_EPS; angle += angle_range.step) {

				ShapeInfo info;
				info.angle = angle;
				info.scale = scale;
				shape_infos.push_back(info);
			}
		}
		return shape_infos;
	}

	Mat KcgMatch::Transform(Mat src, float angle, float scale) {

		Mat dst;
		Point center(src.cols / 2, src.rows / 2);
		Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
		warpAffine(src, dst, rot_mat, src.size());
		return dst;
	}

	Mat KcgMatch::MdlOf(Mat model, ShapeInfo info) {

		return Transform(model, info.angle, info.scale);
	}

	Mat KcgMatch::MskOf(Mat mask, ShapeInfo info) {

		return (Transform(mask, info.angle, info.scale) > 0);
	}

	void KcgMatch::DrawTemplate(Mat& image, Template templ, Scalar color) {

		for (int i = 0; i < templ.features.size(); i++) {

			auto feature = templ.features[i];
			line(image,
				Point(templ.x + feature.x, templ.y + feature.y),
				Point(templ.x + feature.x, templ.y + feature.y),
				color, 1);
		}
	}

	void KcgMatch::QuantifyEdge(Mat image, Mat& angle, Mat& quantized_angle, Mat& mag, float mag_thresh, bool calc_180) {

		Mat dx, dy;
		//Mat dx1, dy1;
		/*Sobel(image, dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		Sobel(image, dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);*/
		float mask_x[3][3] = { { -1,0,1 },{ -2,0,2 },{ -1,0,1 } };
		float mask_y[3][3] = { { 1,2,1 },{ 0,0,0 },{ -1,-2,-1 } };
		Mat kernel_x = Mat(3, 3, CV_32F, mask_x);
		Mat kernel_y = Mat(3, 3, CV_32F, mask_y);
		filter2D(image, dx, CV_32F, kernel_x);
		filter2D(image, dy, CV_32F, kernel_y);
		dx = abs(dx);
		dy = abs(dy);
		mag = dx.mul(dx) + dy.mul(dy);
		phase(dx, dy, angle, true);

		if (calc_180)
			Quantify180(angle, quantized_angle, mag, mag_thresh);
		else
			Quantify8(angle, quantized_angle, mag, mag_thresh);
	}

	void KcgMatch::Quantify8(Mat angle, Mat& quantized_angle, Mat mag, float mag_thresh) {

		Mat_<unsigned char> quantized_unfiltered;
		angle.convertTo(quantized_unfiltered, CV_8U, 16.0f / 360.0f);
        #pragma omp parallel for num_threads(omp_get_num_procs())  // 自动获取CPU核心数
		for (int r = 0; r < angle.rows; ++r)
		{
			unsigned char* quant_ptr = quantized_unfiltered.ptr<unsigned char>(r);
			for (int c = 0; c < angle.cols; ++c)
			{
				quant_ptr[c] &= 7;
			}
		}
		//quantized_unfiltered.copyTo(quantized_angle);
		quantized_angle = Mat::zeros(angle.size(), CV_8U);
		for (int r = 0; r < quantized_angle.rows; ++r) {

			quantized_angle.ptr<unsigned char>(r)[0] = 255;
			quantized_angle.ptr<unsigned char>(r)[quantized_angle.cols - 1] = 255;
		}
		for (int c = 0; c < quantized_angle.cols; ++c) {

			quantized_angle.ptr<unsigned char>(0)[c] = 255;
			quantized_angle.ptr<unsigned char>(quantized_angle.rows - 1)[c] = 255;
		}

		for (int r = 1; r < angle.rows - 1; ++r)
		{
			float* mag_ptr = mag.ptr<float>(r);
			for (int c = 1; c < angle.cols - 1; ++c)
			{
				/*if (mag_ptr[c] >= (mag_thresh * mag_thresh))
				{*/
					int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

					unsigned char* patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += quantized_unfiltered.step1();
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += quantized_unfiltered.step1();
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					// Find bin with the most votes from the patch
					int max_votes = 0;
					int index = -1;
					for (int i = 0; i < 8; ++i)
					{
						if (max_votes < histogram[i])
						{
							index = i;
							max_votes = histogram[i];
						}
					}

					// Only accept the quantization if majority of pixels in the patch agree
					static const int NEIGHBOR_THRESHOLD = 5;
					//if (max_votes >= NEIGHBOR_THRESHOLD)
						quantized_angle.at<unsigned char>(r, c) = index;
					/*else
						quantized_angle.at<unsigned char>(r, c) = 255;*/
				/*}
				else
				{
					quantized_angle.at<unsigned char>(r, c) = 255;
				}*/
			}
		}
	}

	void KcgMatch::Quantify180(Mat angle, Mat& quantized_angle, Mat mag, float mag_thresh) {

		quantized_angle = Mat::zeros(angle.size(), CV_8U);
        #pragma omp parallel for num_threads(omp_get_num_procs())  // 自动获取CPU核心数
		for (int r = 0; r < angle.rows; ++r)
		{
			unsigned char* quantized_angle_ptr = quantized_angle.ptr<unsigned char>(r);
			float* angle_ptr = angle.ptr<float>(r);
			float* mag_ptr = mag.ptr<float>(r);
			for (int c = 0; c < angle.cols; ++c)
			{
				quantized_angle_ptr[c] = (int)round(angle_ptr[c]) % 180;
				/*if (mag_ptr[c] >= (mag_thresh * mag_thresh))
					quantized_angle_ptr[c] = (int)round(angle_ptr[c]) % 180;
				else
					quantized_angle_ptr[c] = 255;*/
			}
		}
	}

	//Template KcgMatch::ExtractTemplate(Mat angle, Mat quantized_angle, Mat mag, ShapeInfo shape_info,
	//	PyramidLevel pl, float weak_thresh, float strong_thresh, int num_features, Mat mask, Mat edge) {

	//	Mat local_angle = Mat(angle.size(), angle.type());
	//	//暂时未用到
	//	//for (int r = 0; r < angle.rows; ++r) {

	//	//	float* angle_ptr = angle.ptr<float>(r);
	//	//	float* local_angle_ptr = local_angle.ptr<float>(r);
	//	//	for (int c = 0; c < angle.cols; ++c) {

	//	//		float dir = angle_ptr[c];
	//	//		if ((dir > 0. && dir < 22.5) || (dir > 157.5 && dir < 202.5) || (dir > 337.5 && dir < 360.))
	//	//			local_angle_ptr[c] = 0.f;
	//	//		else if ((dir > 22.5 && dir < 67.5) || (dir > 202.5 && dir < 247.5))
	//	//			local_angle_ptr[c] = 45.f;
	//	//		else if ((dir > 67.5 && dir < 112.5) || (dir > 247.5 && dir < 292.5))
	//	//			local_angle_ptr[c] = 90.f;
	//	//		else if ((dir > 112.5 && dir < 157.5) || (dir > 292.5 && dir < 337.5))
	//	//			local_angle_ptr[c] = 135.f;
	//	//		else
	//	//			local_angle_ptr[c] = 0.f;
	//	//	}
	//	//}

	//	vector<Candidate> candidates;
	//	candidates.clear();
	//	bool no_mask = mask.empty();
	//	float weak_sq = weak_thresh * weak_thresh;
	//	float strong_sq = strong_thresh * strong_thresh;
	//	float pre_grad, lst_grad;
	//	//暂时未用到
	//	//for (int r = 1; r < mag.rows - 1; ++r)
	//	//{
	//	//	const unsigned char* mask_ptr = no_mask ? NULL : mask.ptr<unsigned char>(r);
	//	//	const float* pre_ptr = mag.ptr<float>(r - 1);
	//	//	const float* cur_ptr = mag.ptr<float>(r);
	//	//	const float* lst_ptr = mag.ptr<float>(r + 1);
	//	//	float* local_angle_ptr = local_angle.ptr<float>(r);

	//	//	for (int c = 1; c < mag.cols - 1; ++c)
	//	//	{
	//	//		if (no_mask || mask_ptr[c])
	//	//		{
	//	//			switch ((int)local_angle_ptr[c]) {

	//	//			case 0:
	//	//				pre_grad = cur_ptr[c - 1];
	//	//				lst_grad = cur_ptr[c + 1];
	//	//				break;
	//	//			case 45:
	//	//				pre_grad = pre_ptr[c + 1];
	//	//				lst_grad = lst_ptr[c - 1];
	//	//				break;
	//	//			case 90:
	//	//				pre_grad = pre_ptr[c];
	//	//				lst_grad = lst_ptr[c];
	//	//				break;
	//	//			case 135:
	//	//				pre_grad = pre_ptr[c - 1];
	//	//				lst_grad = lst_ptr[c + 1];
	//	//				break;
	//	//			}
	//	//			if ((cur_ptr[c] > pre_grad) && (cur_ptr[c] > lst_grad)) {

	//	//				float score = cur_ptr[c];
	//	//				bool validity = false;
	//	//				if (score >= weak_sq) {

	//	//					if (score >= strong_sq) {

	//	//						validity = true;
	//	//					}
	//	//					else {

	//	//						if (((pre_ptr[c - 1]) >= strong_sq) ||
	//	//							((pre_ptr[c]) >= strong_sq) ||
	//	//							((pre_ptr[c + 1]) >= strong_sq) ||
	//	//							((cur_ptr[c - 1]) >= strong_sq) ||
	//	//							((cur_ptr[c + 1]) >= strong_sq) ||
	//	//							((lst_ptr[c - 1]) >= strong_sq) ||
	//	//							((lst_ptr[c]) >= strong_sq) ||
	//	//							((lst_ptr[c + 1]) >= strong_sq))
	//	//						{
	//	//							validity = true;
	//	//						}
	//	//					}
	//	//				}
	//	//				/*if (validity == true &&
	//	//					quantized_angle.at<unsigned char>(r, c) != 255) {

	//	//					Candidate cd;
	//	//					cd.score = score;
	//	//					cd.feature.x = c;
	//	//					cd.feature.y = r;
	//	//					cd.feature.lbl = quantized_angle.at<unsigned char>(r, c);
	//	//					candidates.push_back(cd);
	//	//				}*/
	//	//			}

	//	//		}
	//	//	}
	//	//}

	//	// 1. 创建腐蚀核，缩小 4 个像素
	//	int shrinkPixels = 4;
	//	Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));

	//	// 2. 腐蚀 mask，缩小白色区域
	//	Mat erodedMask;
	//	erode(mask, erodedMask, kernel);

	//	// 3. 与 edge 做与操作，去掉缩小区域的边缘
	//	bitwise_and(edge, erodedMask, edge);

	//	for (int r = 0; r < edge.rows; ++r) {
	//		const uchar* edge_ptr = edge.ptr<uchar>(r);
	//		for (int c = 0; c < edge.cols; ++c) {
	//			if (edge_ptr[c] > 0) {
	//				// 是边缘点，提取对应梯度幅值和方向
	//				float g = mag.at<float>(r, c);          // 幅值
	//				float a = angle.at<float>(r, c);        // 方向，单位：度

	//				//// 可选：将角度量化成 0/45/90/135（作为标签）
	//				//unsigned char lbl;
	//				//if ((a > 0. && a < 22.5) || (a > 157.5 && a < 202.5) || (a > 337.5 && a < 360.))
	//				//	lbl = 0;
	//				//else if ((a > 22.5 && a < 67.5) || (a > 202.5 && a < 247.5))
	//				//	lbl = 45;
	//				//else if ((a > 67.5 && a < 112.5) || (a > 247.5 && a < 292.5))
	//				//	lbl = 90;
	//				//else if ((a > 112.5 && a < 157.5) || (a > 292.5 && a < 337.5))
	//				//	lbl = 135;
	//				//else
	//				//	lbl = 0;

	//				
	//				if (quantized_angle.at<unsigned char>(r, c) != 255 && quantized_angle.at<unsigned char>(r, c) != 0)
	//				{
	//					Candidate cd;
	//					cd.score = g;          // 存梯度幅值
	//					cd.feature.x = c;
	//					cd.feature.y = r;
	//					cd.feature.lbl = quantized_angle.at<unsigned char>(r, c);  // 存方向标签
	//					candidates.push_back(cd);
	//				}
	//					
	//			}
	//		}
	//	}
	//	Template templ;
	//	templ.shape_info.angle = shape_info.angle;
	//	templ.shape_info.scale = shape_info.scale;
	//	templ.pyramid_level = pl;
	//	templ.is_valid = 0;
	//	templ.features.clear();

	//	if (candidates.size() >= num_features && num_features > 0) {

	//		std::stable_sort(candidates.begin(), candidates.end());
	//		float distance = static_cast<float>(candidates.size() / num_features + 1);
	//		templ = SelectScatteredFeatures(candidates, num_features, distance);
	//	}
	//	else {

	//		for (int c = 0; c < candidates.size(); c++) {

	//			templ.features.push_back(candidates[c].feature);
	//		}
	//	}

	//	if (templ.features.size() > 0) {

	//		templ.is_valid = 1;
	//		templ.w = edge.cols;
	//		templ.h = edge.rows;
	//		templ.x = 0;
	//		templ.y =0;
	//		//CropTemplate(templ);
	//	}

	//	return templ;
	//}
	//加速方法1，并行
	Template KcgMatch::ExtractTemplate(Mat angle, Mat quantized_angle, Mat mag, ShapeInfo shape_info,
		PyramidLevel pl, float weak_thresh, float strong_thresh, int num_features, Mat mask, Mat edge) {

		Mat local_angle = Mat(angle.size(), angle.type());
		vector<Candidate> candidates;
		candidates.clear();
		bool no_mask = mask.empty();
		float weak_sq = weak_thresh * weak_thresh;
		float strong_sq = strong_thresh * strong_thresh;
		float pre_grad, lst_grad;
		// 1. 创建腐蚀核，缩小 4 个像素
		int shrinkPixels = 4;
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));

		// 2. 腐蚀 mask，缩小白色区域
		Mat erodedMask;
		erode(mask, erodedMask, kernel);

		// 3. 与 edge 做与操作，去掉缩小区域的边缘
		bitwise_and(edge, erodedMask, edge);
        #pragma omp for schedule(dynamic)  // 动态分配行，负载均衡
		for (int r = 0; r < edge.rows; ++r) {
			const uchar* edge_ptr = edge.ptr<uchar>(r);
			for (int c = 0; c < edge.cols; ++c) {
				if (edge_ptr[c] > 0) {
					// 是边缘点，提取对应梯度幅值和方向
					float g = mag.at<float>(r, c);          // 幅值
					float a = angle.at<float>(r, c);        // 方向，单位：度
					if (quantized_angle.at<unsigned char>(r, c) != 255 && quantized_angle.at<unsigned char>(r, c) != 0)
					{
						Candidate cd;
						cd.score = g;          // 存梯度幅值
						cd.feature.x = c;
						cd.feature.y = r;
						cd.feature.lbl = quantized_angle.at<unsigned char>(r, c);  // 存方向标签
						candidates.push_back(cd);
					}

				}
			}
		}
		Template templ;
		templ.shape_info.angle = shape_info.angle;
		templ.shape_info.scale = shape_info.scale;
		templ.pyramid_level = pl;
		templ.is_valid = 0;
		templ.features.clear();

		if (candidates.size() >= num_features && num_features > 0) {

			//std::stable_sort(candidates.begin(), candidates.end());
			std::sort(std::execution::par, candidates.begin(), candidates.end());
			float distance = static_cast<float>(candidates.size() / num_features + 1);
			templ = SelectScatteredFeatures(candidates, num_features, distance);
		}
		else {

			for (int c = 0; c < candidates.size(); c++) {

				templ.features.push_back(candidates[c].feature);
			}
		}

		if (templ.features.size() > 0) {

			//templ.is_valid = 1;
			//templ.w = edge.cols;
			//templ.h = edge.rows;
			//templ.x = 0;
			//templ.y = 0;
			CropTemplate(templ);
		}

		return templ;
	}
	//加速方法2：opencl
//	// // OpenCL错误检查宏
//#define CL_CHECK(err) \
//    if (err != CL_SUCCESS) { \
//        std::cerr << "OpenCL error at line " << __LINE__ << ": " << err << std::endl; \
//        exit(EXIT_FAILURE); \
//    }
//	// // OpenCL内核：并行提取候选点
//	const char* candidateKernelSource = R"(
//    __kernel void extractCandidates(
//        __global const uchar* edge,         // 边缘图像
//        __global const float* mag,          // 梯度幅值
//        __global const uchar* quantized_angle,  // 量化角度
//        int rows,                           // 图像高度
//        int cols,                           // 图像宽度
//        __global int* candidate_count,      // 候选点计数器（原子操作）
//        __global float* scores,             // 候选点分数（输出）
//        __global int* x_coords,             // 候选点x坐标（输出）
//        __global int* y_coords,             // 候选点y坐标（输出）
//        __global uchar* labels              // 候选点标签（输出）
//    ) {
//        // 计算全局索引（对应像素坐标）
//        int r = get_global_id(0);  // 行
//        int c = get_global_id(1);  // 列
//        
//        // 检查是否在图像范围内
//        if (r >= rows || c >= cols) return;
//        
//        // 计算一维索引（边缘图像）
//        int edge_idx = r * cols + c;
//        
//        // 仅处理边缘点
//        if (edge[edge_idx] > 0) {
//            // 提取梯度幅值和量化角度
//            float g = mag[edge_idx];
//            uchar lbl = quantized_angle[edge_idx];
//            
//            // 筛选有效候选点（排除255和0）
//            if (lbl != 255 && lbl != 0) {
//                // 原子操作获取当前候选点位置并递增计数器
//                int pos = atomic_inc(candidate_count);
//                
//                // 存储候选点信息
//                scores[pos] = g;
//                x_coords[pos] = c;
//                y_coords[pos] = r;
//                labels[pos] = lbl;
//            }
//        }
//    }
//)";
//	// OpenCL加速的ExtractTemplate函数
//	Template KcgMatch::ExtractTemplate(
//		cv::Mat angle, cv::Mat quantized_angle, cv::Mat mag,
//		ShapeInfo shape_info, PyramidLevel pl,
//		float weak_thresh, float strong_thresh,
//		int num_features, cv::Mat mask, cv::Mat edge) {
//
//		// 1. 预处理：形态学操作（保持不变）
//		cv::Mat local_angle = angle.clone();
//		int shrinkPixels = 4;
//		cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT,
//			cv::Size(2 * shrinkPixels + 1, 2 * shrinkPixels + 1));
//		cv::Mat erodedMask;
//		if (!mask.empty()) {
//			cv::erode(mask, erodedMask, kernel1);
//			cv::bitwise_and(edge, erodedMask, edge);  // 过滤边缘
//		}
//
//		int rows = edge.rows;
//		int cols = edge.cols;
//		size_t max_candidates = rows * cols;  // 最大可能候选点数量（边缘点总数）
//
//
//		// 2. OpenCL初始化
//		cl_int err;
//
//		// 获取平台和设备
//		cl_platform_id platform;
//		err = clGetPlatformIDs(1, &platform, nullptr);
//		CL_CHECK(err);
//
//		cl_device_id device;
//		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
//		if (err != CL_SUCCESS) {
//			std::cout << "未找到GPU，使用CPU设备" << std::endl;
//			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
//			CL_CHECK(err);
//		}
//
//		// 创建上下文和命令队列
//		cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
//		CL_CHECK(err);
//
//		cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
//		CL_CHECK(err);
//
//
//		// 3. 准备输入数据并传输到GPU
//		// 输入图像需转换为连续内存（OpenCV默认连续）
//		cl_mem d_edge = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//			rows * cols * sizeof(uchar), edge.data, &err);
//		CL_CHECK(err);
//
//		cl_mem d_mag = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//			rows * cols * sizeof(float), mag.data, &err);
//		CL_CHECK(err);
//
//		cl_mem d_quantized_angle = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//			rows * cols * sizeof(uchar), quantized_angle.data, &err);
//		CL_CHECK(err);
//
//
//		// 4. 创建输出缓冲区（存储候选点）
//		cl_mem d_candidate_count = clCreateBuffer(context, CL_MEM_READ_WRITE,
//			sizeof(int), nullptr, &err);
//		CL_CHECK(err);
//
//		cl_mem d_scores = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
//			max_candidates * sizeof(float), nullptr, &err);
//		CL_CHECK(err);
//
//		cl_mem d_x_coords = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
//			max_candidates * sizeof(int), nullptr, &err);
//		CL_CHECK(err);
//
//		cl_mem d_y_coords = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
//			max_candidates * sizeof(int), nullptr, &err);
//		CL_CHECK(err);
//
//		cl_mem d_labels = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
//			max_candidates * sizeof(uchar), nullptr, &err);
//		CL_CHECK(err);
//
//		// 初始化候选点计数器为0
//		int initial_count = 0;
//		err = clEnqueueWriteBuffer(queue, d_candidate_count, CL_TRUE, 0,
//			sizeof(int), &initial_count, 0, nullptr, nullptr);
//		CL_CHECK(err);
//
//
//		// 5. 编译并执行内核
//		cl_program program = clCreateProgramWithSource(context, 1, &candidateKernelSource, nullptr, &err);
//		CL_CHECK(err);
//
//		// 编译内核（添加优化选项）
//		err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
//		if (err != CL_SUCCESS) {
//			size_t logSize;
//			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
//			std::vector<char> log(logSize);
//			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
//			std::cerr << "内核编译错误:\n" << log.data() << std::endl;
//			exit(EXIT_FAILURE);
//		}
//
//		cl_kernel kernel = clCreateKernel(program, "extractCandidates", &err);
//		CL_CHECK(err);
//
//		// 设置内核参数
//		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_edge);
//		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mag);
//		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_quantized_angle);
//		err |= clSetKernelArg(kernel, 3, sizeof(int), &rows);
//		err |= clSetKernelArg(kernel, 4, sizeof(int), &cols);
//		err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_candidate_count);
//		err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_scores);
//		err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_x_coords);
//		err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_y_coords);
//		err |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_labels);
//		CL_CHECK(err);
//
//		// 配置2D工作项（每个像素一个工作项）
//		size_t globalWorkSize[2] = { (size_t)rows, (size_t)cols };
//		size_t localWorkSize[2] = { 16, 16 };  // 16x16工作组（适配多数GPU）
//
//		// 确保全局工作大小是局部工作大小的整数倍
//		for (int i = 0; i < 2; ++i) {
//			globalWorkSize[i] = ((globalWorkSize[i] + localWorkSize[i] - 1) / localWorkSize[i]) * localWorkSize[i];
//		}
//
//		// 执行内核
//		err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
//			globalWorkSize, localWorkSize,
//			0, nullptr, nullptr);
//		CL_CHECK(err);
//		clFinish(queue);  // 等待内核执行完成
//
//
//		// 6. 从GPU读取结果
//		// 先读取候选点数量
//		int candidate_count = 0;
//		err = clEnqueueReadBuffer(queue, d_candidate_count, CL_TRUE, 0,
//			sizeof(int), &candidate_count, 0, nullptr, nullptr);
//		CL_CHECK(err);
//
//		// 读取候选点数据
//		std::vector<Candidate> candidates;
//		if (candidate_count > 0 && candidate_count <= max_candidates) {
//			candidates.resize(candidate_count);
//			std::vector<float> scores(candidate_count);
//			std::vector<int> x_coords(candidate_count);
//			std::vector<int> y_coords(candidate_count);
//			std::vector<uchar> labels(candidate_count);
//
//			err = clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0,
//				candidate_count * sizeof(float), scores.data(), 0, nullptr, nullptr);
//			CL_CHECK(err);
//
//			err = clEnqueueReadBuffer(queue, d_x_coords, CL_TRUE, 0,
//				candidate_count * sizeof(int), x_coords.data(), 0, nullptr, nullptr);
//			CL_CHECK(err);
//
//			err = clEnqueueReadBuffer(queue, d_y_coords, CL_TRUE, 0,
//				candidate_count * sizeof(int), y_coords.data(), 0, nullptr, nullptr);
//			CL_CHECK(err);
//
//			err = clEnqueueReadBuffer(queue, d_labels, CL_TRUE, 0,
//				candidate_count * sizeof(uchar), labels.data(), 0, nullptr, nullptr);
//			CL_CHECK(err);
//
//			// 组装候选点
//			for (int i = 0; i < candidate_count; ++i) {
//				candidates[i].score = scores[i];
//				candidates[i].feature.x = x_coords[i];
//				candidates[i].feature.y = y_coords[i];
//				candidates[i].feature.lbl = labels[i];
//			}
//		}
//
//
//		// 7. 释放OpenCL资源
//		clReleaseKernel(kernel);
//		clReleaseProgram(program);
//		clReleaseMemObject(d_edge);
//		clReleaseMemObject(d_mag);
//		clReleaseMemObject(d_quantized_angle);
//		clReleaseMemObject(d_candidate_count);
//		clReleaseMemObject(d_scores);
//		clReleaseMemObject(d_x_coords);
//		clReleaseMemObject(d_y_coords);
//		clReleaseMemObject(d_labels);
//		clReleaseCommandQueue(queue);
//		clReleaseContext(context);
//
//
//		// 8. 候选点排序与模板构建（保持原有逻辑）
//		Template templ;
//		templ.shape_info.angle = shape_info.angle;
//		templ.shape_info.scale = shape_info.scale;
//		templ.pyramid_level = pl;
//		templ.is_valid = false;
//		templ.features.clear();
//
//		if (candidates.size() >= num_features && num_features > 0) {
//			// 并行排序（CPU多线程）
//			std::sort(std::execution::par, candidates.begin(), candidates.end());
//			float distance = static_cast<float>(candidates.size() / num_features + 1);
//			templ = SelectScatteredFeatures(candidates, num_features, distance);
//		}
//		else {
//			for (const auto& cd : candidates) {
//				templ.features.push_back(cd.feature);
//			}
//		}
//
//		if (!templ.features.empty()) {
//			templ.is_valid = true;
//			templ.w = edge.cols;
//			templ.h = edge.rows;
//			templ.x = 0;
//			templ.y = 0;
//		}
//
//		return templ;
//	}
	Template KcgMatch::SelectScatteredFeatures(vector<Candidate> candidates, int num_features, float distance) {

		Template templ;
		templ.features.clear();
		float distance_sq = distance * distance;
		int i = 0;
		while (templ.features.size() < num_features) {

			Candidate c = candidates[i];
			// Add if sufficient distance away from any previously chosen feature
			bool keep = true;
			for (int j = 0; (j < (int)templ.features.size()) && keep; ++j)
			{
				Feature f = templ.features[j];
				keep = ((c.feature.x - f.x) * (c.feature.x - f.x) + (c.feature.y - f.y) * (c.feature.y - f.y) >= distance_sq);
			}
			if (keep)
				templ.features.push_back(c.feature);

			if (++i == (int)candidates.size())
			{
				// Start back at beginning, and relax required distance
				i = 0;
				distance -= 1.0f;
				distance_sq = distance * distance;
				// if (distance < 3)
				// {
				//     // we don't want two features too close
				//     break;
				// }
			}
		}
		return templ;
	}

	Rect KcgMatch::CropTemplate(Template& templ) {

		int min_x = std::numeric_limits<int>::max();
		int min_y = std::numeric_limits<int>::max();
		int max_x = std::numeric_limits<int>::min();
		int max_y = std::numeric_limits<int>::min();

		// First pass: find min/max feature x,y 
		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			int x = templ.features[i].x;
			int y = templ.features[i].y;
			min_x = std::min(min_x, x);
			min_y = std::min(min_y, y);
			max_x = std::max(max_x, x);
			max_y = std::max(max_y, y);
		}

		/// @todo Why require even min_x, min_y?
		if (min_x % 2 == 1)
			--min_x;
		if (min_y % 2 == 1)
			--min_y;

		// Second pass: set width/height and shift all feature positions
		templ.w = (max_x - min_x)+1;
		templ.h = (max_y - min_y)+1;
		templ.x = min_x;
		templ.y = min_y;

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			templ.features[i].x -= templ.x;
			templ.features[i].y -= templ.y;
		}
		return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	}

	void KcgMatch::LoadRegion8Idxes() {

		int keys[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
		region8_idxes_.clear();
		int angle_region = (int)((angle_range_.end - angle_range_.begin) / angle_range_.step) + 1;
		int scale_region = (int)((scale_range_.end - scale_range_.begin) / scale_range_.step) + 1;
		for (int ar = 0; ar < angle_region; ar++) {

			float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
			if (cur_agl < 0.f) cur_agl += 360.f;
			int idx = 0;
			for (int i = 0; i < 16; i++) {

				if (cur_agl >= AngleRegionTable[i][0] &&
					cur_agl < AngleRegionTable[i][1]) {

					idx = i;
					break;
				}
			}
			if (keys[idx] == 0) {

				for (int sr = 0; sr < scale_region; sr++) {

					region8_idxes_.push_back(ar + sr * angle_region);
				}
			}
			keys[idx] = 1;
		}
	}

	void KcgMatch::SaveModel() {

		int total_templ = 0;
		for (int i = 0; i < PyramidLevel_TabooUse; i++) {

			total_templ += (int)templ_all_[i].size();
		}
		assert((total_templ / PyramidLevel_TabooUse) == templ_all_[0].size());
		int match_range_size = (int)templ_all_[0].size();
		string model_name = model_root_ + class_name_ + KCG_MODEL_SUFFUX;
		FileStorage fs(model_name, FileStorage::WRITE);
		fs << "class_name" << class_name_;
		fs << "total_pyramid_levels" << PyramidLevel_7;
		fs << "angle_range_bgin" << angle_range_.begin;
		fs << "angle_range_end" << angle_range_.end;
		fs << "angle_range_step" << angle_range_.step;
		fs << "scale_range_bgin" << scale_range_.begin;
		fs << "scale_range_end" << scale_range_.end;
		fs << "scale_range_step" << scale_range_.step;
		fs << "templates"
			<< "[";
		{
			for (int i = 0; i < match_range_size; i++) {

				fs << "{";
				fs << "template_id" << int(i);
				fs << "template_pyrds"
					<< "[";
				{
					for (int j = 0; j < PyramidLevel_TabooUse; j++) {

						auto templ = templ_all_[j][i];
						fs << "{";
						fs << "id" << int(i);
						fs << "pyramid_level" << templ.pyramid_level;
						fs << "is_valid" << templ.is_valid;
						fs << "x" << templ.x;
						fs << "y" << templ.y;
						fs << "w" << templ.w;
						fs << "h" << templ.h;
						fs << "shape_scale" << templ.shape_info.scale;
						fs << "shape_angle" << templ.shape_info.angle;
						fs << "feature_size" << (int)templ.features.size();
						fs << "features"
							<< "[";
						{
							for (int k = 0; k < (int)templ.features.size(); k++) {

								auto feat = templ.features[k];
								fs << "[:" << feat.x << feat.y << feat.lbl << "]";
							}
						}
						fs << "]";
						// 存储 edge_features 信息
						fs << "edge_feature_size" << (int)templ.edge_features.size();
						fs << "edge_features" << "[";
						for (const auto& ef : templ.edge_features) {
							fs << "[:" << ef.pt.x << ef.pt.y << ef.grad_x << ef.grad_y << "]";
						}
						fs << "]";
						fs << "}";
					}
				}
				fs << "]";
				fs << "}";
			}
		}
		fs << "]";
	}
	//采用多线程
	void KcgMatch::SaveModel_fast() {
		int total_templ = 0;
		for (int i = 0; i < PyramidLevel_TabooUse; i++) {
			total_templ += (int)templ_all_[i].size();
		}
		assert((total_templ / PyramidLevel_TabooUse) == templ_all_[0].size());
		int match_range_size = (int)templ_all_[0].size();
		std::string model_name = model_root_ + class_name_ + KCG_MODEL_SUFFUX;

		// 第一阶段：写入模型元数据
		{
			FileStorage fs(model_name, FileStorage::WRITE);
			fs << "class_name" << class_name_;
			fs << "total_pyramid_levels" << PyramidLevel_7;
			fs << "angle_range_bgin" << angle_range_.begin;
			fs << "angle_range_end" << angle_range_.end;
			fs << "angle_range_step" << angle_range_.step;
			fs << "scale_range_bgin" << scale_range_.begin;
			fs << "scale_range_end" << scale_range_.end;
			fs << "scale_range_step" << scale_range_.step;

			// 开始写入模板数组
			fs << "templates" << "[";
			// 不关闭方括号，将在后续阶段由主线程关闭
		}

		// 第二阶段：多线程写入模板数据
		std::mutex fileMutex;
		std::vector<std::thread> threads;

		// 为每个角度模板创建一个线程
		for (int i = 0; i < match_range_size; i++) {
			threads.emplace_back([this, i, match_range_size, &fileMutex, &model_name]() {
				// 打开文件追加模式，使用互斥锁保护整个写入过程
				std::lock_guard<std::mutex> lock(fileMutex);
				FileStorage fs(model_name, FileStorage::APPEND);

				// 写入当前角度模板的头部
				fs << "{";
				fs << "template_id" << int(i);
				fs << "template_pyrds" << "[";

				// 写入该角度在各金字塔层级的模板
				for (int j = 0; j < PyramidLevel_TabooUse; j++) {
					auto templ = templ_all_[j][i];
					fs << "{";
					fs << "id" << int(i);
					fs << "pyramid_level" << templ.pyramid_level;
					fs << "is_valid" << templ.is_valid;
					fs << "x" << templ.x;
					fs << "y" << templ.y;
					fs << "w" << templ.w;
					fs << "h" << templ.h;
					fs << "shape_scale" << templ.shape_info.scale;
					fs << "shape_angle" << templ.shape_info.angle;
					fs << "feature_size" << (int)templ.features.size();

					// 写入特征点
					fs << "features" << "[";
					for (int k = 0; k < (int)templ.features.size(); k++) {
						auto feat = templ.features[k];
						fs << "[:" << feat.x << feat.y << feat.lbl << "]";
					}
					fs << "]";

					// 写入边缘特征
					fs << "edge_feature_size" << (int)templ.edge_features.size();
					fs << "edge_features" << "[";
					for (const auto& ef : templ.edge_features) {
						fs << "[:" << ef.pt.x << ef.pt.y << ef.grad_x << ef.grad_y << "]";
					}
					fs << "]";

					fs << "}";
				}

				// 写入该角度模板的尾部
				fs << "]";
				fs << "}";

				// 文件存储对象在作用域结束时自动释放
				});
		}

		// 等待所有线程完成
		for (auto& t : threads) {
			t.join();
		}

		// 第三阶段：关闭模板数组
		{
			FileStorage fs(model_name, FileStorage::APPEND);
			fs << "]"; // 关闭 "templates" 数组
		}
	}
	void KcgMatch::SaveModel1() {
		int total_templ = 0;
		for (int i = 0; i < PyramidLevel_TabooUse; i++) {
			total_templ += (int)templ_all_[i].size();
		}

		assert((total_templ / PyramidLevel_TabooUse) == templ_all_[0].size());
		int match_range_size = (int)templ_all_[0].size();

		string model_name = model_root_ + class_name_ + KCG_MODEL_SUFFUX;
		FileStorage fs(model_name, FileStorage::WRITE);

		fs << "class_name" << class_name_;
		fs << "total_pyramid_levels" << PyramidLevel_7;
		fs << "angle_range_bgin" << angle_range_.begin;
		fs << "angle_range_end" << angle_range_.end;
		fs << "angle_range_step" << angle_range_.step;
		fs << "scale_range_bgin" << scale_range_.begin;
		fs << "scale_range_end" << scale_range_.end;
		fs << "scale_range_step" << scale_range_.step;

		fs << "templates" << "[";
		for (int i = 0; i < match_range_size; i++) {
			fs << "{";
			fs << "template_id" << int(i);
			fs << "template_pyrds" << "[";

			for (int j = 0; j < PyramidLevel_TabooUse; j++) {
				const auto& templ = templ_all_[j][i];

				fs << "{";
				fs << "id" << int(i);
				fs << "pyramid_level" << templ.pyramid_level;
				fs << "is_valid" << templ.is_valid;
				fs << "x" << templ.x;
				fs << "y" << templ.y;
				fs << "w" << templ.w;
				fs << "h" << templ.h;
				fs << "shape_scale" << templ.shape_info.scale;
				fs << "shape_angle" << templ.shape_info.angle;

				// 存储 feature 信息
				fs << "feature_size" << (int)templ.features.size();
				fs << "features" << "[";
				for (const auto& feat : templ.features) {
					fs << "[:" << feat.x << feat.y << feat.lbl << "]";
				}
				fs << "]";

				// 存储 small_edge_features 信息
				fs << "small_edge_features_size" << (int)templ.small_edge_features.size();
				fs << "small_edge_features" << "[";
				for (const auto& ef : templ.small_edge_features) {
					fs << "[:" << ef.pt.x << ef.pt.y << ef.grad_x << ef.grad_y << "]";
				}
				fs << "]";
				// 存储 edge_features 信息
				fs << "edge_feature_size" << (int)templ.edge_features.size();
				fs << "edge_features" << "[";
				for (const auto& ef : templ.edge_features) {
					fs << "[:" << ef.pt.x << ef.pt.y << ef.grad_x << ef.grad_y << "]";
				}
				fs << "]";

				fs << "}";
			}

			fs << "]"; // end of template_pyrds
			fs << "}"; // end of template_id block
		}
		fs << "]"; // end of templates
	}


	void KcgMatch::LoadModel() {

		ClearModel();
		string model_name = model_root_ + class_name_ + KCG_MODEL_SUFFUX;
		FileStorage fs(model_name, FileStorage::READ);
		assert(fs.isOpened() && "load model failed.");
		FileNode fn = fs.root();
		angle_range_.begin = fn["angle_range_bgin"];
		angle_range_.end = fn["angle_range_end"];
		angle_range_.step = fn["angle_range_step"];
		scale_range_.begin = fn["scale_range_bgin"];
		scale_range_.end = fn["scale_range_end"];
		scale_range_.step = fn["scale_range_step"];
		FileNode tps_fn = fn["templates"];
		FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
		for (; tps_it != tps_it_end; ++tps_it)
		{
			int template_id = (*tps_it)["template_id"];
			FileNode pyrds_fn = (*tps_it)["template_pyrds"];
			FileNodeIterator pyrd_it = pyrds_fn.begin(), pyrd_it_end = pyrds_fn.end();
			int pl = 0;
			for (; pyrd_it != pyrd_it_end; ++pyrd_it)
			{
				FileNode pyrd_fn = (*pyrd_it);
				Template templ;
				templ.id = pyrd_fn["id"];
				templ.pyramid_level = pyrd_fn["pyramid_level"];
				templ.is_valid = pyrd_fn["is_valid"];
				templ.x = pyrd_fn["x"];
				templ.y = pyrd_fn["y"];
				templ.w = pyrd_fn["w"];
				templ.h = pyrd_fn["h"];
				templ.shape_info.scale = pyrd_fn["shape_scale"];
				templ.shape_info.angle = pyrd_fn["shape_angle"];
				FileNode features_fn = pyrd_fn["features"];
				FileNodeIterator feature_it = features_fn.begin(), feature_it_end = features_fn.end();
				for (; feature_it != feature_it_end; ++feature_it)
				{
					FileNode feature_fn = (*feature_it);
					FileNodeIterator feature_info = feature_fn.begin();
					Feature feat;
					feature_info >> feat.x >> feat.y >> feat.lbl;
					templ.features.push_back(feat);
				}
				// === 加载 edge_features ===
				FileNode edge_features_fn = pyrd_fn["edge_features"];
				FileNodeIterator feature_it1 = edge_features_fn.begin(), feature_it_end1 = edge_features_fn.end();
				for (; feature_it1 != feature_it_end1; ++feature_it1) {
					FileNode edge_features_fn = (*feature_it1);
					FileNodeIterator feature_info = edge_features_fn.begin();
					EdgePointWithGradient epg;
					double x, y;
					float gx, gy;
					feature_info >> x >> y >> gx >> gy;
					epg.pt = cv::Point2f(x, y);
					epg.grad_x = gx;
					epg.grad_y = gy;
					templ.edge_features.push_back(epg);
				}
				templ_all_[pl].push_back(templ);
				pl++;
			}
		}

		LoadRegion8Idxes();
	}

	void KcgMatch::LoadModel1() {
		ClearModel();

		string model_name = model_root_ + class_name_ + KCG_MODEL_SUFFUX;
		FileStorage fs(model_name, FileStorage::READ);
		assert(fs.isOpened() && "load model failed.");

		FileNode fn = fs.root();

		angle_range_.begin = fn["angle_range_bgin"];
		angle_range_.end = fn["angle_range_end"];
		angle_range_.step = fn["angle_range_step"];
		scale_range_.begin = fn["scale_range_bgin"];
		scale_range_.end = fn["scale_range_end"];
		scale_range_.step = fn["scale_range_step"];

		FileNode tps_fn = fn["templates"];
		FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();

		for (; tps_it != tps_it_end; ++tps_it) {
			int template_id = (*tps_it)["template_id"];
			FileNode pyrds_fn = (*tps_it)["template_pyrds"];
			FileNodeIterator pyrd_it = pyrds_fn.begin(), pyrd_it_end = pyrds_fn.end();

			int pl = 0;
			for (; pyrd_it != pyrd_it_end; ++pyrd_it) {
				FileNode pyrd_fn = (*pyrd_it);
				Template templ;

				templ.id = pyrd_fn["id"];
				templ.pyramid_level = pyrd_fn["pyramid_level"];
				templ.is_valid = pyrd_fn["is_valid"];
				templ.x = pyrd_fn["x"];
				templ.y = pyrd_fn["y"];
				templ.w = pyrd_fn["w"];
				templ.h = pyrd_fn["h"];
				templ.shape_info.scale = pyrd_fn["shape_scale"];
				templ.shape_info.angle = pyrd_fn["shape_angle"];
				// === 加载 features ===
				FileNode features_fn = pyrd_fn["features"];
				FileNodeIterator feature_it = features_fn.begin(), feature_it_end = features_fn.end();
				for (; feature_it != feature_it_end; ++feature_it) {
					FileNode feature_fn = (*feature_it);
					FileNodeIterator feature_info = feature_fn.begin();
					Feature feat;
					feature_info >> feat.x >> feat.y >> feat.lbl;
					templ.features.push_back(feat);
				}
				// === 加载 samll edge_features ===
				FileNode small_edge_features_fn = pyrd_fn["small_edge_features"];
				FileNodeIterator feature_it_samll = small_edge_features_fn.begin(), feature_it_end_samll = small_edge_features_fn.end();
				for (; feature_it_samll != feature_it_end_samll; ++feature_it_samll) {
					FileNode edge_features_fn = (*feature_it_samll);
					FileNodeIterator feature_info = edge_features_fn.begin();
					smallEdgePointWithGradient epg;
					int x, y;
					float gx, gy;
					feature_info >> x >> y >> gx >> gy;
					epg.pt = cv::Point(x, y);
					epg.grad_x = gx;
					epg.grad_y = gy;
					templ.small_edge_features.push_back(epg);
				}
				// === 加载 edge_features ===
				FileNode edge_features_fn = pyrd_fn["edge_features"];
				FileNodeIterator feature_it1 = edge_features_fn.begin(), feature_it_end1 = edge_features_fn.end();
				for (; feature_it1 != feature_it_end1; ++feature_it1) {
					FileNode edge_features_fn = (*feature_it1);
					FileNodeIterator feature_info = edge_features_fn.begin();
					EdgePointWithGradient epg;
					int x, y;
					float gx, gy;
					feature_info >> x >> y >> gx >> gy;
					epg.pt = cv::Point(x, y);
					epg.grad_x = gx;
					epg.grad_y = gy;
					templ.edge_features.push_back(epg);
				}

				templ_all_[pl].push_back(templ);
				pl++;
			}
		}

		//LoadRegion8Idxes();
	}


	void KcgMatch::ClearModel() {

		for (int i = 0; i < PyramidLevel_TabooUse; i++) {

			templ_all_[i].clear();
		}
	}

	void KcgMatch::InitMatchParameter(float score_thresh, float overlap, float mag_thresh, float greediness, int T, int top_k, MatchingStrategy strategy) {

		score_thresh_ = score_thresh;
		overlap_ = overlap;
		mag_thresh_ = mag_thresh;
		greediness_ = greediness;
		T_ = T;
		top_k_ = top_k;
		strategy_ = strategy;
	}

	void KcgMatch::GetAllPyramidLevelValidSource(cv::Mat& source, PyramidLevel pyrd_level) {

		sources_.clear();
		for (int pl = 0; pl <= pyrd_level; pl++) {

			Mat source_pyrd;
			if (pl == 0) source_pyrd = source;
			else pyrDown(source, source_pyrd, Size(source.cols >> 1, source.rows >> 1));
			source = source_pyrd;
			sources_.push_back(source_pyrd);
		}
	}

	/*vector<Match> KcgMatch::GetTopKMatches(vector<Match> matches) {

		vector<Match> top_k_matches;
		top_k_matches.clear();
		if (top_k_ > 0 && (top_k_ < matches.size()) && (matches.size() > 0)) {

			int k = 0;
			top_k_matches.push_back(matches[0]);
			for (int m = 1; m < matches.size(); m++) {

				if (matches[m].similarity < matches[m - 1].similarity) {

					++k;
					if (k >= top_k_) break;
				}
				top_k_matches.push_back(matches[m]);
			}
		}
		else
		{
			top_k_matches = matches;
		}
		return top_k_matches;
	}*/
	vector<Match> KcgMatch::GetTopKMatches(vector<Match> matches) {
		vector<Match> top_k_matches;
		if (matches.empty())
			return top_k_matches;

		// 找到 similarity 最大的那个 match
		auto max_it = std::max_element(matches.begin(), matches.end(),
			[](const Match& a, const Match& b) {
				return a.similarity < b.similarity;
			});

		top_k_matches.push_back(*max_it);
		return top_k_matches;
	}


	vector<Match> KcgMatch::DoNmsMatches(vector<Match> matches, PyramidLevel pl, float overlap) {

		vector<Rect> boxes; boxes.clear();
		vector<float> scores; scores.clear();
		vector<int> indices; indices.clear();
		for (int m = 0; m < matches.size(); m++) {

			auto templ = templ_all_[pl][matches[m].template_id];
			Rect box = Rect(matches[m].x, matches[m].y, templ.w, templ.h);
			boxes.insert(boxes.end(), box);
			scores.insert(scores.end(), matches[m].similarity);
		}
		cv_dnn_nms::NMSBoxes(boxes, scores, overlap, overlap, indices);
		vector<Match> final_matches; final_matches.clear();
		for (auto index : indices) {

			final_matches.push_back(matches[index]);
		}
		return final_matches;
	}

	vector<Match> KcgMatch::MatchingPyrd180(Mat src, PyramidLevel pl, vector<int> region_idxes) {

		pl = PyramidLevel(pl + 3);
		vector<Match> matches; matches.clear();
		Mat angle, quantized_angle, mag;
		QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, true);
		score_thresh_ = 0.7;
		vector<int> cout_sum;
		vector<double> all_score;
#pragma omp parallel 
		{
			int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
#pragma omp for nowait
			for (int t = 0; t < tlsz; t++) {

				Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
				for (int r = 0; r < quantized_angle.rows - templ.h; r++) {

					for (int c = 0; c < quantized_angle.cols - templ.w; c++) {

						int fsz = (int)templ.features.size();
						float partial_sum = 0.f;
						bool valid = true;
						int count = 0;
						for (int f = 0; f < fsz; f++) {

							Feature feat = templ.features[f];
							int sidx = quantized_angle.ptr<unsigned char>(r + feat.y)[c + feat.x];
							int tidx = feat.lbl;
							if (sidx != 255) {

								partial_sum += score_table_[sidx][tidx];
								
							}
							else
							{
								count++;
							}
							//|| partial_sum + (fsz - f) * greediness_ < score_thresh_ * fsz
							if (count > fsz * 0.2) {

								valid = false;
								break;
							}
						}
						
						if (valid) {

							float score = partial_sum / fsz;
							if (score >=score_thresh_) {

								Match match;
								match.x = c;
								match.y = r;
								match.similarity = score;
								match.template_id = templ.id;
								//score_thresh_ = score;
#pragma omp critical
								matches.insert(matches.end(), match);
								cout_sum.push_back(count);
							}
							
						}

					}
				}
				
			}
			
			
			cv::Mat grad_x, grad_y, source_pyrd, grad_x_pyrd, grad_y_pyrd;
			Sobel(src, grad_x, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
			Sobel(src, grad_y, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
			for (int i = 0; i < matches.size(); i++)
			{
				Template templ = templ_all_[pl][matches[i].template_id];
				float totalSim = 0.0f;
				int validCount = 0;
				double bestscore = 0;
				for (const auto& ep : templ.edge_features) {
					int px = matches[i].x + ep.pt.x;
					int py = matches[i].y + ep.pt.y;

					float src_gradx = grad_x.at<float>(py, px);
					float src_grady = grad_y.at<float>(py, px);
					float magSrc = std::sqrt(src_gradx * src_gradx + src_grady * src_grady);
					float gnx = src_gradx / magSrc;
					float gny = src_grady / magSrc;
					float magTemplate = std::sqrt(ep.grad_x * ep.grad_x + ep.grad_y * ep.grad_y);
					//if (magTemplate > 1e-9) {  // 只考虑非零梯度的像素
					float gtx = ep.grad_x / magTemplate;
					float gty = ep.grad_y / magTemplate;
					totalSim += gnx * gtx + gny * gty;
					validCount++;
					//}
				}
				float avgSim = totalSim / validCount;
				bestscore = avgSim;
				all_score.push_back(bestscore);
				//if (validCount > 0) {
				//	float avgSim = totalSim / validCount;
				//	if (avgSim > bestscore) { // 可调节粗匹配阈值
				//		bestscore = avgSim;
				//		all_score.push_back(bestscore);
				//	}
				//}

			}
		}
		
		matches = DoNmsMatches(matches, pl, overlap_);
		return matches;
	}
	vector<Match> KcgMatch::MatchingPyrd180(Mat src, PyramidLevel pl,  vector<Match> matches, vector<int> region_idxes) {
		int zoom = pl;
		pl = PyramidLevel(3);
		vector<Match> refined_matches;
		Mat angle, quantized_angle, mag;
		QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, true);
		score_thresh_ = 0.6;
		vector<int> cout_sum;
		vector<double> all_score;
#pragma omp parallel
		{
			int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
#pragma omp for nowait
			for (int t = 0; t < tlsz; t++) {

				Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);

				for (size_t m = 0; m < matches.size(); m++) {
					
					int base_r = matches[m].y * ((zoom == 0) ? 1 : (zoom * 2));
					int base_c = matches[m].x * ((zoom == 0) ? 1 : (zoom * 2));

					// 遍历 ±20 区域
					for (int r = max(0, base_r - 20); r <= min(quantized_angle.rows - templ.h, base_r + 20); r++) {
						for (int c = max(0, base_c - 20); c <= min(quantized_angle.cols - templ.w, base_c + 20); c++) {

							int fsz = (int)templ.features.size();
							float partial_sum = 0.f;
							bool valid = true;
							int count = 0;

							for (int f = 0; f < fsz; f++) {

								Feature feat = templ.features[f];
								int sidx = quantized_angle.ptr<unsigned char>(r + feat.y)[c + feat.x];
								int tidx = feat.lbl;

								if (sidx != 255&& sidx >= 0 && sidx < 180 && tidx >= 0 && tidx < 180) {
									partial_sum += score_table_[sidx][tidx];
								}
								else {
									count++;
								}

								if (count > fsz * 0.2) {
									valid = false;
									break;
								}
							}

							if (valid) {

								float score = partial_sum / fsz;
								if (score >= score_thresh_) {

									Match match;
									match.x = c;
									match.y = r;
									match.similarity = score;
									match.template_id = templ.id;
									match.r = templ.shape_info.angle;

#pragma omp critical
									refined_matches.push_back(match);
									cout_sum.push_back(count);
								}
							}

						}
					}

				}
			}
		}

		matches = DoNmsMatches(refined_matches, pl, overlap_);
		return matches;
	}

	vector<Match> KcgMatch::MatchingPyrd8(Mat src, PyramidLevel pl, vector<int> region_idxes) {

		vector<Match> matches; matches.clear();
		Mat angle, quantized_angle, mag;
		QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, false);
		Mat spread_angle;
		Spread(quantized_angle, spread_angle, T_);
		vector<Mat> response_maps;
		ComputeResponseMaps(spread_angle, response_maps);
#pragma omp parallel 
		{
			int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
#pragma omp for nowait
			for (int t = 0; t < tlsz; t++) {

				Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
				for (int r = 0; r < quantized_angle.rows - templ.h; r += T_) {

					for (int c = 0; c < quantized_angle.cols - templ.w; c += T_) {

						int fsz = (int)templ.features.size();
						int partial_sum = 0;
						bool valid = true;
						for (int f = 0; f < fsz; f++) {

							Feature feat = templ.features[f];
							int label = feat.lbl;
							partial_sum +=
								response_maps[label].ptr<unsigned char>(r + feat.y)[c + feat.x];
							if (partial_sum + (fsz - f) * greediness_ < score_thresh_ * fsz) {

								valid = false;
								break;
							}
						}
						
						if (valid) {

							float score = partial_sum / (100.f * fsz);
							if (score >= score_thresh_) {

								Match match;
								match.x = c;
								match.y = r;
								match.similarity = score;
								match.template_id = templ.id;
#pragma omp critical
								matches.insert(matches.end(), match);
							}
						}
					}
				}
			}
		}
		matches = DoNmsMatches(matches, pl, overlap_);
		return matches;
	}

	void KcgMatch::Spread(const Mat quantized_angle, Mat& spread_angle, int T) {

		spread_angle = Mat::zeros(quantized_angle.size(), CV_8U);
		int cols = quantized_angle.cols;
		int rows = quantized_angle.rows;
		int half_T = 0;
		if (T != 1) half_T = T / 2;
#pragma omp parallel for
		for (int r = half_T; r < rows - half_T; r++) {

			for (int c = half_T; c < cols - half_T; c++) {

				for (int i = -half_T; i <= half_T; i++) {

					for (int j = -half_T; j <= half_T; j++) {

						unsigned char shift_bits =
							quantized_angle.ptr<unsigned char>(r + i)[c + j];
						if (shift_bits < 8) {

							spread_angle.ptr<unsigned char>(r)[c] |=
								(unsigned char)(1 << shift_bits);
						}
					}
				}
			}
		}
	}

	void KcgMatch::ComputeResponseMaps(const Mat spread_angle, vector<Mat>& response_maps) {

		response_maps.clear();
		for (int i = 0; i < 8; i++) {

			Mat rm;
			rm.create(spread_angle.size(), CV_8U);
			response_maps.push_back(rm);
		}
		int cols = spread_angle.cols;
		int rows = spread_angle.rows;
#pragma omp parallel for
		for (int i = 0; i < 8; i++) {

			for (int r = 0; r < rows; r++) {

				for (int c = 0; c < cols; c++) {

					response_maps[i].ptr<unsigned char>(r)[c] =
						score_table_8map_[i][spread_angle.ptr<unsigned char>(r)[c]];
				}
			}
		}
	}

	bool KcgMatch::CalcPyUpRoiAndStartPoint(PyramidLevel cur_pl, PyramidLevel obj_pl, Match match,
		Mat& r, Point& p, bool is_padding) {

		auto templ = templ_all_[cur_pl][match.template_id];
		int padding = 0;
		if (is_padding) {

			int min_side = std::min(templ.w, templ.h);
			int diagonal_line_length = (int)ceil(sqrt(templ.w * templ.w + templ.h * templ.h));
			padding = diagonal_line_length - min_side;
		}
		int err_pl = cur_pl - obj_pl;
		int T = 2 * T_;
		int extend_pixel = 1;
		cv::Point bp, ep;
		int multiple = (1 << err_pl);
		match.x -= (T + padding) / 2;
		match.y -= (T + padding) / 2;
		templ.w += (T + padding);
		templ.h += (T + padding);
		bp.x = (match.x - extend_pixel) * multiple;
		bp.y = (match.y - extend_pixel) * multiple;
		ep.x = (match.x + templ.w + extend_pixel) * multiple;
		ep.y = (match.y + templ.h + extend_pixel) * multiple;
		if (bp.x < 0) bp.x = 0;
		if (bp.y < 0) bp.y = 0;
		if (ep.x < 0) ep.x = 0;
		if (ep.y < 0) ep.y = 0;
		if (bp.x >= sources_[obj_pl].cols) bp.x = sources_[obj_pl].cols - 1;
		if (bp.y >= sources_[obj_pl].rows) bp.y = sources_[obj_pl].rows - 1;
		if (ep.x >= sources_[obj_pl].cols) ep.x = sources_[obj_pl].cols - 1;
		if (ep.y >= sources_[obj_pl].rows) ep.y = sources_[obj_pl].rows - 1;
		if (bp.x != ep.x || bp.y != ep.y) {

			Rect rect = Rect(bp, ep);
			//Mat roi(sources_[obj_pl], rect);
			Mat roi(sources_[obj_pl]);
			r = roi;
			p = bp;
			return true;
		}
		else
		{
			return false;
		}
	}

	void KcgMatch::CalcRegionIndexes(vector<int>& region_idxes, Match match, MatchingStrategy strategy) {

		region_idxes.clear();
		Template templ = templ_all_[PyramidLevel_0][match.template_id];
		float match_agl = templ.shape_info.angle;
		float match_sal = templ.shape_info.scale;
		int angle_region = (int)((angle_range_.end - angle_range_.begin) / angle_range_.step) + 1;
		int scale_region = (int)((scale_range_.end - scale_range_.begin) / scale_range_.step) + 1;
		if (strategy <= Strategy_Middling) {

			if (match_agl < 0.f) match_agl += 360.f;
			int key = (int)floor(match_agl / 22.5f);
			float left_agl = match_agl - key * 22.5f;
			for (int ar = 0; ar < angle_region; ar++) {

				float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
				if (cur_agl < 0.f) cur_agl += 360.f;
				int k = key;
				if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {

					for (int sr = 0; sr < scale_region; sr++) {

						region_idxes.push_back(ar + sr * angle_region);
					}
				}
				if (strategy == Strategy_Accurate) {

					if (left_agl < 11.25f) {

						k = key - 1;
						if (k < 0) k = 15;
						if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {

							for (int sr = 0; sr < scale_region; sr++) {

								region_idxes.push_back(ar + sr * angle_region);
							}
						}
					}
					else
					{
						k = key + 1;
						if (k > 15) k = 0;
						if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {

							for (int sr = 0; sr < scale_region; sr++) {

								region_idxes.push_back(ar + sr * angle_region);
							}
						}
					}
				}
			}
		}
		else if (strategy == Strategy_Rough) {

			float err_range = 3.f;
			for (int ar = 0; ar < angle_region; ar++) {

				float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
				if (cur_agl >= (match_agl - angle_range_.step * err_range) &&
					cur_agl <= (match_agl + angle_range_.step * err_range)) {

					for (int sr = 0; sr < scale_region; sr++) {

						float cur_sal = templ_all_[PyramidLevel_0][ar + sr * angle_region].shape_info.scale;
						if (cur_sal >= (match_sal - scale_range_.step * err_range) &&
							cur_sal <= (match_sal + scale_range_.step * err_range)) {

							region_idxes.push_back(ar + sr * angle_region);
						}
					}
				}
			}
		}
	
	}

	vector<Match> KcgMatch::ReconfirmMatches(vector<Match> matches, PyramidLevel pl) {

		vector<Match> rf_matches;
		rf_matches.clear();
		for (int i = 0; i < matches.size(); i++) {

			Mat roi;
			Point sp;
			CalcPyUpRoiAndStartPoint(pl, pl, matches[i], roi, sp, true);
			vector<int> region_idxes;
			CalcRegionIndexes(region_idxes, matches[i], Strategy_Accurate);
			auto tmp_matches = MatchingPyrd8(roi, pl, region_idxes);
			if (tmp_matches.size() > 0) {

				tmp_matches[0].x += sp.x;
				tmp_matches[0].y += sp.y;
				rf_matches.push_back(tmp_matches[0]);
			}
		}
		rf_matches = DoNmsMatches(rf_matches, pl, overlap_);
		return rf_matches;
	}

	vector<Match> KcgMatch::MatchingFinal(vector<Match> matches, PyramidLevel pl) {

		vector<Match> final_matches;
		final_matches.clear();
		for (int i = 0; i < matches.size(); i++) {

			Mat roi;
			Point sp;
			CalcPyUpRoiAndStartPoint(pl, PyramidLevel_0, matches[i], roi, sp, true);
			vector<int> region_idxes;
			CalcRegionIndexes(region_idxes, matches[i], strategy_);
			if(matches[i].template_id>0)
			    region_idxes.push_back(matches[i].template_id - 1);
			region_idxes.push_back(matches[i].template_id);
			region_idxes.push_back(matches[i].template_id + 1);
			//region_idxes.push_back(matches[i].template_id+1);
			//auto tmp_matches = MatchingPyrd180(roi, PyramidLevel_0, region_idxes);
			auto tmp_matches = MatchingPyrd180(roi, pl, matches, region_idxes);
			if (tmp_matches.size() > 0) {

				/*tmp_matches[0].x += sp.x;
				tmp_matches[0].y += sp.y;*/
				final_matches.push_back(tmp_matches[0]);
			}
		}
		final_matches = DoNmsMatches(final_matches, pl, overlap_);
		return final_matches;
	}
	int KcgMatch::addTemplate_rotate(int p, double theta)
	{
		Template templ;
		templ.pyramid_level = p;
		Candidate cd;
		int l = 1;
		int template_id = 1;
		cv::Point2f center;
		center.x = templ_all_[p][0].x + templ_all_[p][0].w / 2.0f;
		center.y = templ_all_[p][0].y + templ_all_[p][0].h / 2.0f;
		//std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
		//int template_id = static_cast<int>(template_pyramids.size());
		//const auto& to_rotate_tp = template_pyramids[zero_id];
		//TemplatePyramid tp;
		//tp.resize(pyramid_levels);
		/*for (int l = 0; l < pyramid_levels; ++l)
		{*/
			//if (p > 0) center /= 2;
			templ.shape_info.angle = theta;
			templ.shape_info.scale = 1;
			int num_bins = 8;
			float angle_per_bin = 180.0f / num_bins;
			std::set<std::pair<int, int>> coord_set;
			for (auto& f : templ_all_[p][0].features)
			{
				Point2f p_first;
				p_first.x = f.x + templ_all_[p][0].x;
				p_first.y = f.y + templ_all_[p][0].y;
				double angleRad = theta / 180 * M_PI;
				Point2f p_rot = rotatePoint(p_first, center, angleRad);
				Feature f_new;
				f_new.x = int(p_rot.x + 0.5f);
				f_new.y = int(p_rot.y + 0.5f);
				
				// 正确的方向旋转和重新量化
				float original_angle = f.lbl * angle_per_bin;
				float rotated_angle = original_angle - theta;

				while (rotated_angle < 0) rotated_angle += 180.0f;
				while (rotated_angle >= 180.0f) rotated_angle -= 180.0f;

				f_new.lbl = static_cast<int>(rotated_angle / angle_per_bin + 0.5f) % num_bins;
				auto coord = std::make_pair(f_new.x, f_new.y);
				if (coord_set.find(coord) == coord_set.end()) {
					coord_set.insert(coord);
					templ.features.push_back(f_new);
				}
				
			}
			//f_new.theta = f.theta - theta;
				/*while (f_new.theta > 360) f_new.theta -= 360;
				while (f_new.theta < 0) f_new.theta += 360;*/
			if (templ.features.size() > 0) {

				templ.is_valid = 1;
				CropTemplate(templ);
			}
			templ_all_[p].push_back(templ);
			// ===== 处理 p+3 层模板 =====
			Template templ2;  // 使用新模板对象，避免特征混入
			templ2.pyramid_level = p ;
			templ2.shape_info.angle = theta;
			templ2.shape_info.scale = 1;
			center.x = templ_all_[p+3][0].x + templ_all_[p+3][0].w / 2.0f;
			center.y = templ_all_[p+3][0].y + templ_all_[p+3][0].h / 2.0f;
			for (auto& f : templ_all_[p+3][0].features)
			{
				Point2f p_first;
				p_first.x = f.x + templ_all_[p + 3][0].x;
				p_first.y = f.y + templ_all_[p + 3][0].y;
				Point2f p_rot = rotatePoint(p_first, center, -theta / 180 * CV_PI);
				Feature f_new;
				f_new.x = int(p_rot.x + 0.5f);
				f_new.y = int(p_rot.y + 0.5f);
				//修改
				// 计算旋转后的方向（考虑模板旋转角度）
				float rotated_lbl = f.lbl + theta;

				// 规范化到0-179度范围
				// 1. 先将角度调整到0-359.99度
				while (rotated_lbl < 0.0f) rotated_lbl += 360.0f;
				while (rotated_lbl >= 360.0f) rotated_lbl -= 360.0f;

				// 2. 映射到0-179度（180个bin）
				if (rotated_lbl >= 180.0f) {
					rotated_lbl -= 180.0f;
				}
				// 存储规范化后的方向
				f_new.lbl = rotated_lbl;
				//f_new.theta = f.theta - theta;
				/*while (f_new.theta > 360) f_new.theta -= 360;
				while (f_new.theta < 0) f_new.theta += 360;*/
				//原始
		/*		float theta_rot = theta;
				f_new.lbl = f.lbl - std::fmod(theta_rot, 180.0f);*/
				templ2.features.push_back(f_new);
			}
			if (templ2.features.size() > 0) {

				templ2.is_valid = 1;
				CropTemplate(templ2);
			}
			templ_all_[p + 3].push_back(templ2);
			//tp[l].pyramid_level = l;
		//}
		
			//cropTemplates(tp);
		//template_pyramids.push_back(tp);
		return template_id;
	}
	static cv::Point2f rotate2d(const cv::Point2f inPoint, const double & angRad)
	{
		cv::Point2f outPoint;
		//CW rotation
		outPoint.x = std::cos(angRad) * inPoint.x - std::sin(angRad) * inPoint.y;
		outPoint.y = std::sin(angRad) * inPoint.x + std::cos(angRad) * inPoint.y;
		return outPoint;
	}
	 cv::Point2f  KcgMatch::rotatePoint(const cv::Point2f inPoint, const cv::Point2f center, const double  angRad)
	{
		return rotate2d(inPoint - center, angRad) + center;
	}
	static Rect cropTemplates(std::vector<Template>& templates)
	{
		int min_x = std::numeric_limits<int>::max();
		int min_y = std::numeric_limits<int>::max();
		int max_x = std::numeric_limits<int>::min();
		int max_y = std::numeric_limits<int>::min();
		// First pass: find min/max feature x,y over all pyramid levels and modalities
		//    这里的templates.size()其实就等于金字塔的层数
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template& templ = templates[i];
			for (int j = 0; j < (int)templ.features.size(); ++j)
			{
				//    在原始模板图像上对应的位置
				int x = templ.features[j].x << templ.pyramid_level;
				int y = templ.features[j].y << templ.pyramid_level;
				min_x = std::min(min_x, x);
				min_y = std::min(min_y, y);
				max_x = std::max(max_x, x);
				max_y = std::max(max_y, y);
			}
		}
		//    以上代码得到了所有模板特征点在原图中的最大和最小位置
		if (min_x % 2 == 1)
			--min_x;
		if (min_y % 2 == 1)
			--min_y;

		//    校正下位置
		// Second pass: set width/height and shift all feature positions
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template& templ = templates[i];
			templ.w = (max_x - min_x) >> templ.pyramid_level;
			templ.h = (max_y - min_y) >> templ.pyramid_level;
			templ.x = min_x >> templ.pyramid_level;
			templ.y = min_y >> templ.pyramid_level;
			for (int j = 0; j < (int)templ.features.size(); ++j)
			{
				templ.features[j].x -= templ.x;
				templ.features[j].y -= templ.y;
			}
		}
		return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	}
	
} // end namespace kcg_matching

