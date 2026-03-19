	/************************************** 点云配准 *****************************************************/
// 初始化参数
	cam_lift_ = cam_lift;
	cam_right_ = cam_right;
	camera_liftt = camera_lift;
	camera_rightt = camera_right;
	point_data_liftt = point_data_lift;
	point_data_rightt = point_data_right;
	rgb_data_liftt = rgb_data_lift;
	rgb_data_rightt = rgb_data_right;
	gray_data_liftt = gray_data_lift;
	gray_data_rightt = gray_data_right;
	leftFeatures.clear();
	rightFeatures.clear();
	leftPoints.clear();
	rightPoints.clear();

	//// ********** 点云分割出人脸 ********** 

	int width2 = 686;
	int height2 = 952;

	int lx = 1025;
	int ly = 731;
	int rx = 776;
	int ry = 701;

	Eigen::Matrix4f ICP_r;
	ICP_r << 0.99951, 0.0287052, 0.0127156, -8.22554,
	-0.0312582, 0.87209, 0.488343, -327.071,
	-0.00292825, -0.488499, 0.872565, 89.3636,
	0, 0, 0, 1;

	//// ********** 点云分割出人脸 ********** 


			//增加冗余空间
	float padding = 120.0f;
	lx = lx - padding;
	ly = ly - padding;
	rx = rx - padding;
	ry = ry - padding;
	width2 = width2 + 2 * padding;
	height2 = height2 + 2 * padding;

	// 提取目标图像
	cv::Mat targetImage3 = imread("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/rgb_face_roi_l.png");
	cv::Mat targetImage4 = imread("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/rgb_face_roi_r.png");

	// 加载68点模型
	dlib::shape_predictor shapePredictor;
	dlib::deserialize("C:/Users/tester/Desktop/FaceStitche/shape_predictor_68_face_landmarks.dat") >> shapePredictor;

	// Mat 转 dlib 图像
	dlib::cv_image<dlib::bgr_pixel> dlibLeftImage(targetImage3);
	dlib::cv_image<dlib::bgr_pixel> dlibRightImage(targetImage4);

	// 检测器
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

	// 检测人脸
	std::vector<dlib::rectangle> leftDetections = detector(dlibLeftImage);
	std::vector<dlib::rectangle> rightDetections = detector(dlibRightImage);

	// 提取左图 ROI（左眼、嘴唇）
	if (!leftDetections.empty()) {
		dlib::full_object_detection shape = shapePredictor(dlibLeftImage, leftDetections[0]);
		leftEyeROI = extractROI(shape, { 36,37,38,39,40,41 }); // 左眼
		lipsROI = extractROI(shape, { 48,49,50,51,52,53,54,55,56,57,58,59,
										 60,61,62,63,64,65,66,67 }); // 嘴唇
	}

	// 提取右图 ROI（右眼）
	if (!rightDetections.empty()) {
		dlib::full_object_detection shape = shapePredictor(dlibRightImage, rightDetections[0]);
		rightEyeROI = extractROI(shape, { 42,43,44,45,46,47 });
	}

	//取出点云cloud
	pcl::PolygonMesh mesh;
	pcl::io::loadPLYFile("DataSave/OrgData/mesh.ply", mesh);
	PointCloud::Ptr cloud(new PointCloud);
	pcl::fromPCLPointCloud2(mesh.cloud, *cloud);


	//后面纹理坐标部分的参数
	float rate = 1.0;// 宽保留比
	const int col = height2;
	const int row = width2 * rate * 2;

	// 定义变量保存拼接后的图像
	Mat stitchedImage(height2, width2 * rate * 2, CV_8UC3);
	// 左右矩形各一半（各占 rate）
	Mat leftROI = stitchedImage(Rect(0, 0, width2 * rate, height2));
	Mat rightROI = stitchedImage(Rect(width2 * rate, 0, width2 * rate, height2));
	//提前拼接需要用到的stitchedImage
	targetImage3(Rect(0, 0, width2 * rate, height2)).copyTo(leftROI);
	targetImage4(Rect(width2 * (1.0 - rate), 0, width2 * rate, height2)).copyTo(rightROI);

	// 继续后续的旋转和保存操作...
	stitchedImage = RGB90right(stitchedImage);

	cv::imwrite("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/stitchedImage.png", stitchedImage);




	// 存储所有符合条件的点的索引
	std::vector<size_t> candidate_indices;
	std::vector<size_t> sampled_indices;

	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		const auto& point = cloud->points[i];
		//if (point.y > -95.0 && point.y < -25.0)
		if (point.y > -93.0 && point.y < -25)
		{
			candidate_indices.push_back(i);
		}
	}

	// 左右脸特征点颜色
	std::vector<cv::Vec3b> leftFeatures;
	std::vector<cv::Vec3b> rightFeatures;
	std::vector<cv::Point> leftPoints;
	std::vector<cv::Point> rightPoints;

	struct PointData {
		float x, y, z;
		float left_u, left_v;
		int left_x, left_y;
		float right_u, right_v;
		int right_x, right_y;
	};

	// ------------------ Step1: 在全局范围内随机采样 ------------------
	size_t target_total = 400; // 你之前的总样本目标 (视需要你也可以改成 num_to_sample)
	size_t effective_total = std::min(target_total, candidate_indices.size());

	// 随机发生器（固定种子，结果可复现）
	std::mt19937 gen(14350);

	// 复制一份候选点用于洗牌，避免破坏原本的 candidate_indices
	std::vector<size_t> temp_candidates = candidate_indices;

	// Fisher–Yates 全局随机采样
	if (temp_candidates.size() > effective_total) {
		for (size_t i = 0; i < effective_total; ++i) {
			std::uniform_int_distribution<size_t> dist(i, temp_candidates.size() - 1);
			size_t j = dist(gen);
			std::swap(temp_candidates[i], temp_candidates[j]);
		}
	}

	// 填充到原本的 sampled_indices 容器中 (假设外部已定义，这里先清空再提取前 effective_total 个)
	sampled_indices.clear();
	sampled_indices.assign(temp_candidates.begin(), temp_candidates.begin() + effective_total);

	std::cout << "实际全局抽样数量: " << sampled_indices.size() << std::endl;

	// ------------------ Step2: 收集采样点数据 ------------------
	auto insideImage = [](int x, int y, const cv::Mat& img) -> bool {
		return (x >= 0 && x < img.cols && y >= 0 && y < img.rows);
		};

	std::vector<PointData> currentPoints;
	leftFeatures.clear();
	rightFeatures.clear();
	leftPoints.clear();
	rightPoints.clear();

	for (size_t k = 0; k < sampled_indices.size(); ++k) {
		size_t i = sampled_indices[k]; // 使用采样后的全局索引

		const auto& point = cloud->points[i];

		// 左相机纹理坐标
		PointT liftpoint_left = pointright2lift(point, ICP_r);
		Point2f left_uv = Point2RGBXY(1944, 2592, liftpoint_left, 0);
		float left_u = (left_uv.x - (1944 - height2 - ly)) / col;
		float left_v = 1.00f - (left_uv.y - lx) / row;
		int left_x = static_cast<int>(left_uv.x - (1944 - height2 - ly));
		int left_y = static_cast<int>(left_uv.y - lx);

		// 右相机纹理坐标
		Point2f right_uv = Point2RGBXY(1944, 2592, point, 1);
		float right_u = (right_uv.x - (1944 - height2 - ry)) / col;
		float right_v = 1.00f - ((right_uv.y + width2 * rate - rx)) / row;
		int right_x = static_cast<int>(right_uv.x - (1944 - height2 - ry));
		int right_y = static_cast<int>(right_uv.y + width2 * rate - rx);

		// ===== 严格边界检查：越界点直接舍弃 =====
		if (!insideImage(left_x, left_y, stitchedImage)) {
			continue;
		}
		if (!insideImage(right_x, right_y, stitchedImage)) {
			continue;
		}

		currentPoints.push_back({
			point.x, point.y, point.z,
			left_u, left_v, left_x, left_y,
			right_u, right_v, right_x, right_y
			});

		// 两边颜色：直接取，不再 clamp
		cv::Vec3b left_color = stitchedImage.at<cv::Vec3b>(left_y, left_x);
		leftFeatures.push_back(left_color);
		leftPoints.push_back(cv::Point(left_x, left_y));

		cv::Vec3b right_color = stitchedImage.at<cv::Vec3b>(right_y, right_x);
		rightFeatures.push_back(right_color);
		rightPoints.push_back(cv::Point(right_x, right_y));
	}

	if (leftFeatures.empty() || rightFeatures.empty()) {
		std::cout << "没有有效采样点（严格边界检查后为空）" << std::endl;
		return 0;
	}


	// 计算每个特征点的颜色差异（欧氏距离）
	std::vector<float> colorDiffs;
	colorDiffs.reserve(leftFeatures.size());

	for (size_t i = 0; i < leftFeatures.size(); i++)
	{
		cv::Vec3b left = leftFeatures[i];
		cv::Vec3b right = rightFeatures[i];

		float diff = std::sqrt(
			std::pow(float(left[0]) - float(right[0]), 2.0f) +
			std::pow(float(left[1]) - float(right[1]), 2.0f) +
			std::pow(float(left[2]) - float(right[2]), 2.0f)
		);
		colorDiffs.push_back(diff);
	}

	// 计算颜色差异的均值
	float meanDiff = 0.0f;
	for (float diff : colorDiffs)
	{
		meanDiff += diff;
	}
	meanDiff /= static_cast<float>(colorDiffs.size());

	// 筛选出正常点（差异小于 2 倍均值）
	std::vector<cv::Vec3b> filteredLeftFeatures;
	std::vector<cv::Vec3b> filteredRightFeatures;
	std::vector<cv::Point> filteredLeftPoints;
	std::vector<cv::Point> filteredRightPoints;
	std::vector<int> filteredIndices;

	for (size_t i = 0; i < leftFeatures.size(); i++)
	{
		if (colorDiffs[i] <= 2.0f * meanDiff) {
			filteredLeftFeatures.push_back(leftFeatures[i]);
			filteredRightFeatures.push_back(rightFeatures[i]);
			filteredLeftPoints.push_back(leftPoints[i]);
			filteredRightPoints.push_back(rightPoints[i]);
			filteredIndices.push_back(static_cast<int>(i) + 20);
		}
	}

	if (filteredLeftFeatures.empty()) {
		std::cout << "筛选后没有有效点，无法继续。" << std::endl;
		return 0;
	}

	// 打印筛选结果
	std::cout << "\n特征点颜色差异分析:" << std::endl;
	std::cout << "平均颜色差异: " << meanDiff << std::endl;
	std::cout << "筛选阈值: " << 2.0f * meanDiff << std::endl;
	std::cout << "原始特征点数量: " << leftFeatures.size() << std::endl;
	std::cout << "筛选后特征点数量: " << filteredLeftFeatures.size() << std::endl;
	std::cout << "剔除异常点数量: " << leftFeatures.size() - filteredLeftFeatures.size() << std::endl;


	// ===== 阶段1：矩阵 M 校正 =====
	Eigen::Matrix3f M = compute_color_transform(filteredLeftFeatures, filteredRightFeatures);
	std::cout << "色彩变换矩阵 M:\n" << M << std::endl;

	// 1) 整图乘矩阵
	cv::Mat targetImage3_matrix = apply_color_transform(targetImage3.clone(), M);

	// 2) 逐点生成“矩阵后的左侧颜色”用于评估
	std::vector<cv::Vec3b> transformedFeatures_M;
	transformedFeatures_M.reserve(filteredLeftFeatures.size());

	float meanDiff_M = 0.0f;
	for (size_t i = 0; i < filteredLeftFeatures.size(); ++i) {
		const cv::Vec3b origLeft = filteredLeftFeatures[i];
		const cv::Vec3b targetRight = filteredRightFeatures[i];

		Eigen::Vector3f v(origLeft[0], origLeft[1], origLeft[2]); // BGR
		Eigen::Vector3f u = M * v;

		cv::Vec3b out(
			cv::saturate_cast<uchar>(u[0]),
			cv::saturate_cast<uchar>(u[1]),
			cv::saturate_cast<uchar>(u[2])
		);
		transformedFeatures_M.push_back(out);

		float d = std::sqrt(
			std::pow(float(out[0]) - float(targetRight[0]), 2.0f) +
			std::pow(float(out[1]) - float(targetRight[1]), 2.0f) +
			std::pow(float(out[2]) - float(targetRight[2]), 2.0f)
		);
		meanDiff_M += d;
	}
	meanDiff_M /= static_cast<float>(filteredLeftFeatures.size());
	std::cout << "矩阵变换后平均颜色差: " << meanDiff_M << std::endl;


	// ===========================================================================
	// 保存阶段1拼接结果: stitchedImage2
	// ===========================================================================
	cv::Mat stitchedImage2(height2, width2 * rate * 2, CV_8UC3);
	cv::Mat leftROI2 = stitchedImage2(cv::Rect(0, 0, width2 * rate, height2));
	cv::Mat rightROI2 = stitchedImage2(cv::Rect(width2 * rate, 0, width2 * rate, height2));

	targetImage3_matrix(cv::Rect(0, 0, width2 * rate, height2)).copyTo(leftROI2);
	targetImage4(cv::Rect(width2 * (1.0 - rate), 0, width2 * rate, height2)).copyTo(rightROI2);
	std::cout << ">> 阶段1：矩阵校正RGB图片拼接完成 (stitchedImage2)" << std::endl;

	// 继续后续的旋转和保存操作...
	stitchedImage2 = RGB90right(stitchedImage2);
	cv::imwrite("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/stitchedImage2.png", stitchedImage2);

	// ===========================================================================
// ===== 阶段2：Reinhard 亮度校正 (基于对应点统计量，而不是整张图) =====
// ===========================================================================

	auto getLab = [](cv::Vec3b bgr8) {
		cv::Mat m(1, 1, CV_32FC3);
		m.at<cv::Vec3f>(0, 0) = cv::Vec3f(
			bgr8[0] / 255.0f,
			bgr8[1] / 255.0f,
			bgr8[2] / 255.0f
		);
		cv::cvtColor(m, m, cv::COLOR_BGR2Lab);
		return m.at<cv::Vec3f>(0, 0);
		};

	auto calcMeanStd = [](const std::vector<float>& values, double& mean, double& stddev) {
		mean = 0.0;
		stddev = 0.0;
		if (values.empty()) return;

		for (float v : values) mean += v;
		mean /= static_cast<double>(values.size());

		double var = 0.0;
		for (float v : values) {
			double d = static_cast<double>(v) - mean;
			var += d * d;
		}
		var /= static_cast<double>(values.size());
		stddev = std::sqrt(var);
		};

	// 1) 用“对应点”统计阶段1左图与右图参考图的 L 通道均值/方差
	std::vector<float> L_stage1_points;
	std::vector<float> L_ref_points;
	L_stage1_points.reserve(transformedFeatures_M.size());
	L_ref_points.reserve(filteredRightFeatures.size());

	for (size_t i = 0; i < transformedFeatures_M.size(); ++i) {
		cv::Vec3f lab_s1 = getLab(transformedFeatures_M[i]);
		cv::Vec3f lab_ref = getLab(filteredRightFeatures[i]);

		L_stage1_points.push_back(lab_s1[0]); // L
		L_ref_points.push_back(lab_ref[0]);   // L
	}

	double mu_S = 0.0, sigma_S = 0.0;
	double mu_T = 0.0, sigma_T = 0.0;
	calcMeanStd(L_stage1_points, mu_S, sigma_S);
	calcMeanStd(L_ref_points, mu_T, sigma_T);

	std::cout << "阶段2(L通道对应点统计) mu_S = " << mu_S
		<< ", sigma_S = " << sigma_S
		<< ", mu_T = " << mu_T
		<< ", sigma_T = " << sigma_T << std::endl;

	// 2) 第二阶段明确在第一阶段图像 targetImage3_matrix 上进行
	cv::Mat src32f;
	targetImage3_matrix.convertTo(src32f, CV_32FC3, 1.0 / 255.0);

	cv::Mat labSrc;
	cv::cvtColor(src32f, labSrc, cv::COLOR_BGR2Lab);

	std::vector<cv::Mat> srcCh;
	cv::split(labSrc, srcCh); // srcCh[0] 是 L

	if (sigma_S > 1e-6) {
		srcCh[0] = (srcCh[0] - mu_S) * (sigma_T / sigma_S) + mu_T;
	}
	else {
		// 避免除零：退化为只做均值平移
		srcCh[0] = (srcCh[0] - mu_S) + mu_T;
	}

	// 限制 L 通道范围到 [0, 100]
	cv::min(srcCh[0], 100.0, srcCh[0]);
	cv::max(srcCh[0], 0.0, srcCh[0]);

	cv::merge(srcCh, labSrc);

	cv::Mat targetImage3_reinhard32f;
	cv::cvtColor(labSrc, targetImage3_reinhard32f, cv::COLOR_Lab2BGR);

	cv::Mat targetImage3_reinhard;
	targetImage3_reinhard32f.convertTo(targetImage3_reinhard, CV_8UC3, 255.0);

	// ===========================================================================
	// 保存阶段2拼接结果: stitchedImage3
	// ===========================================================================
	cv::Mat stitchedImage3(height2, width2 * rate * 2, CV_8UC3);
	cv::Mat leftROI3 = stitchedImage3(cv::Rect(0, 0, width2 * rate, height2));
	cv::Mat rightROI3 = stitchedImage3(cv::Rect(width2 * rate, 0, width2 * rate, height2));

	targetImage3_reinhard(cv::Rect(0, 0, width2 * rate, height2)).copyTo(leftROI3);
	targetImage4(cv::Rect(width2 * (1.0 - rate), 0, width2 * rate, height2)).copyTo(rightROI3);
	std::cout << ">> 阶段2：Reinhard亮度校正RGB图片拼接完成 (stitchedImage3)" << std::endl;

	// 继续后续的旋转和保存操作...
	stitchedImage3 = RGB90right(stitchedImage3);
	cv::imwrite("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/stitchedImage3.png", stitchedImage3);

// ===========================================================================
// ===== 指标评估 (统一在“旋转后的拼接图坐标系”中进行) =====
// ===========================================================================

	std::cout << "\n================ 色彩校正阶段性评估 (ΔE_76 & ΔE_RGB) ================\n";

	auto deltaE76 = [](const cv::Vec3f& lab1, const cv::Vec3f& lab2) {
		return std::sqrt(
			std::pow(double(lab1[0]) - double(lab2[0]), 2.0) +
			std::pow(double(lab1[1]) - double(lab2[1]), 2.0) +
			std::pow(double(lab1[2]) - double(lab2[2]), 2.0)
		);
		};

	auto deltaRGB = [](const cv::Vec3b& c1, const cv::Vec3b& c2) {
		return std::sqrt(
			std::pow(double(c1[0]) - double(c2[0]), 2.0) +
			std::pow(double(c1[1]) - double(c2[1]), 2.0) +
			std::pow(double(c1[2]) - double(c2[2]), 2.0)
		);
		};

	auto getMean = [](const std::vector<double>& v) {
		if (v.empty()) return 0.0;
		double s = 0.0;
		for (double d : v) s += d;
		return s / static_cast<double>(v.size());
		};

	auto improve = [](double base, double now) {
		if (std::abs(base) < 1e-12) return 0.0;
		return (base - now) / base * 100.0;
		};

	auto isInside = [](const cv::Mat& img, const cv::Point& p) {
		return p.x >= 0 && p.x < img.cols && p.y >= 0 && p.y < img.rows;
		};

	// 注意：这里要求 stitchedImage / stitchedImage2 / stitchedImage3
	// 都是“已经 RGB90right 之后”的版本，且与 filteredLeftPoints / filteredRightPoints 坐标一致。

	if (!filteredLeftPoints.empty() &&
		filteredLeftPoints.size() == filteredRightPoints.size()) {

		std::vector<double> dE_LAB_Raw, dE_LAB_Stage1, dE_LAB_Stage2;
		std::vector<double> dE_RGB_Raw, dE_RGB_Stage1, dE_RGB_Stage2;

		size_t validEvalCount = 0;

		for (size_t i = 0; i < filteredLeftPoints.size(); ++i) {
			const cv::Point lp = filteredLeftPoints[i];
			const cv::Point rp = filteredRightPoints[i];

			// 所有阶段都在同一个坐标系里检查
			if (!isInside(stitchedImage, lp))  continue;
			if (!isInside(stitchedImage, rp))  continue;
			if (!isInside(stitchedImage2, lp)) continue;
			if (!isInside(stitchedImage3, lp)) continue;

			// 原始阶段：从原始拼接图取
			const cv::Vec3b BGR_Raw = stitchedImage.at<cv::Vec3b>(lp.y, lp.x);
			const cv::Vec3b BGR_Ref = stitchedImage.at<cv::Vec3b>(rp.y, rp.x);

			// 阶段1：从阶段1拼接图取真实像素
			const cv::Vec3b BGR_S1 = stitchedImage2.at<cv::Vec3b>(lp.y, lp.x);

			// 阶段2：从阶段2拼接图取真实像素
			const cv::Vec3b BGR_S2 = stitchedImage3.at<cv::Vec3b>(lp.y, lp.x);

			const cv::Vec3f Lab_Ref = getLab(BGR_Ref);
			const cv::Vec3f Lab_Raw = getLab(BGR_Raw);
			const cv::Vec3f Lab_S1 = getLab(BGR_S1);
			const cv::Vec3f Lab_S2 = getLab(BGR_S2);

			dE_LAB_Raw.push_back(deltaE76(Lab_Raw, Lab_Ref));
			dE_LAB_Stage1.push_back(deltaE76(Lab_S1, Lab_Ref));
			dE_LAB_Stage2.push_back(deltaE76(Lab_S2, Lab_Ref));

			dE_RGB_Raw.push_back(deltaRGB(BGR_Raw, BGR_Ref));
			dE_RGB_Stage1.push_back(deltaRGB(BGR_S1, BGR_Ref));
			dE_RGB_Stage2.push_back(deltaRGB(BGR_S2, BGR_Ref));

			++validEvalCount;
		}

		if (validEvalCount == 0) {
			std::cout << "评估失败：严格边界检查后，没有可用评估点。" << std::endl;
		}
		else {
			double mLabRaw = getMean(dE_LAB_Raw);
			double mLabS1 = getMean(dE_LAB_Stage1);
			double mLabS2 = getMean(dE_LAB_Stage2);

			double mRgbRaw = getMean(dE_RGB_Raw);
			double mRgbS1 = getMean(dE_RGB_Stage1);
			double mRgbS2 = getMean(dE_RGB_Stage2);

			std::cout << "有效评估点数量: " << validEvalCount << std::endl;
			std::cout << "------------------------------------------------------------\n";

			printf("[ΔE_RGB] 原始状态 (Original):    %.4f\n", mRgbRaw);
			printf("[ΔE_RGB] 阶段1 (Matrix):       %.4f (改善: %.2f%%)\n", mRgbS1, improve(mRgbRaw, mRgbS1));
			printf("[ΔE_RGB] 阶段2 (Reinhard):     %.4f (改善: %.2f%%)\n", mRgbS2, improve(mRgbRaw, mRgbS2));

			std::cout << "------------------------------------------------------------\n";

			printf("[ΔE_LAB] 原始状态 (Original):    %.4f\n", mLabRaw);
			printf("[ΔE_LAB] 阶段1 (Matrix):       %.4f (改善: %.2f%%)\n", mLabS1, improve(mLabRaw, mLabS1));
			printf("[ΔE_LAB] 阶段2 (Reinhard):     %.4f (改善: %.2f%%)\n", mLabS2, improve(mLabRaw, mLabS2));

			std::cout << "============================================================\n";
		}
	}


	//// =============== 打印详细的颜色信息（对照论文三阶段逻辑） ===============
	//std::cout << "\n特征点颜色对比 (格式: BGR):\n";
	//// M = Matrix, F = Final (Reinhard + Gamma)
	//std::cout << "索引 | 原始颜色差 | 变换后(M)差 | 最终(F)差   | 左原BGR | 左(M)后BGR | 最终(F)后BGR | 右目标BGR\n";
	//std::cout << "-------------------------------------------------------------------------------------------------------------\n";

	//for (size_t i = 0; i < filteredLeftFeatures.size(); ++i)
	//{
	//	int idx = (i < filteredIndices.size() ? filteredIndices[i] : int(i));
	//	const cv::Vec3b origLeft = filteredLeftFeatures[i];
	//	const cv::Vec3b left_M = transformedFeatures_M[i];
	//	const cv::Vec3b left_F = transformedFeatures_Final[i]; // 修正：使用 Final 标识符
	//	const cv::Vec3b rightRef = filteredRightFeatures[i];

	//	float origDiff = (i < colorDiffs.size()) ? colorDiffs[i] : euclid_bgr(origLeft, rightRef);
	//	const float diff_M = euclid_bgr(left_M, rightRef);
	//	const float diff_F = euclid_bgr(left_F, rightRef);

	//	std::printf("%4d | %12.1f | %12.1f | %12.1f | (%3d,%3d,%3d) | (%3d,%3d,%3d) | (%3d,%3d,%3d) | (%3d,%3d,%3d)\n",
	//		idx, origDiff, diff_M, diff_F,
	//		origLeft[0], origLeft[1], origLeft[2],
	//		left_M[0], left_M[1], left_M[2],
	//		left_F[0], left_F[1], left_F[2],
	//		rightRef[0], rightRef[1], rightRef[2]
	//	);
	//}

	//// =============== ΔE 评估：原始 vs 右、矩阵后 vs 右、最终精修后 vs 右 ===============
	//{
	//	const size_t N = filteredLeftFeatures.size();
	//	if (N > 0) {
	//		std::vector<double> dE76_before, dE76_after_M, dE76_after_Final;
	//		std::vector<double> dE00_before, dE00_after_M, dE00_after_Final;

	//		for (size_t i = 0; i < N; ++i) {
	//			const cv::Vec3f LabL = BGRu8_to_Lab32f(filteredLeftFeatures[i]);   // 原始
	//			const cv::Vec3f LabR = BGRu8_to_Lab32f(filteredRightFeatures[i]);  // 目标
	//			const cv::Vec3f LabLM = BGRu8_to_Lab32f(transformedFeatures_M[i]);  // 阶段1: Matrix
	//			const cv::Vec3f LabLF = BGRu8_to_Lab32f(transformedFeatures_Final[i]); // 阶段2: R+G

	//			dE76_before.push_back(deltaE76(LabL, LabR));
	//			dE76_after_M.push_back(deltaE76(LabLM, LabR));
	//			dE76_after_Final.push_back(deltaE76(LabLF, LabR));

	//			dE00_before.push_back(deltaE2000(LabL, LabR));
	//			dE00_after_M.push_back(deltaE2000(LabLM, LabR));
	//			dE00_after_Final.push_back(deltaE2000(LabLF, LabR));
	//		}

	//		auto print_stats = [](const char* name, const std::vector<double>& v) {
	//			const double m = mean_val(v);
	//			std::cout << name << " -> mean ΔE: " << m << "\n";
	//			};

	//		std::cout << "\n================ ΔE 指标（Lab）评估 (Matrix + Reinhard + Gamma) ================\n";
	//		print_stats("[ΔE76]  原始状态", dE76_before);
	//		print_stats("[ΔE76]  阶段1 (Matrix)", dE76_after_M);
	//		print_stats("[ΔE76]  最终精修 (Final)", dE76_after_Final);
	//		std::cout << "------------------------------------------------\n";

	//		auto rel_improve = [](double a, double b) { return (a > 1e-9) ? (a - b) / a * 100.0 : 0.0; };
	//		std::cout << "总改善率 (Overall Improvement: 原始 -> Final): "
	//			<< rel_improve(mean_val(dE76_before), mean_val(dE76_after_Final)) << "%\n";
	//		std::cout << "===============================================================================\n";
	//	}
	//}

	// 保存结果
	cv::imwrite("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/corrected_left_cpp.png", targetImage3);


	//// 定义变量保存拼接后的图像
	//Mat stitchedImage2(height2, width2 * rate * 2, CV_8UC3);
	//// 左右矩形各一半（各占 rate）
	//Mat leftROI2 = stitchedImage2(Rect(0, 0, width2 * rate, height2));
	//Mat rightROI2 = stitchedImage2(Rect(width2 * rate, 0, width2 * rate, height2));

	//targetImage3(Rect(0, 0, width2 * rate, height2)).copyTo(leftROI2);
	//targetImage4(Rect(width2 * (1.0 - rate), 0, width2 * rate, height2)).copyTo(rightROI2);
	//cout << "rgb图片拼接完成" << endl;

	//// 继续后续的旋转和保存操作...
	//stitchedImage2 = RGB90right(stitchedImage2);
	//cv::imwrite("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/stitchedImage2.png", stitchedImage2);

	cout << "开始处理曲面" << endl;

	/****************入口2，曲面mesh****************/
	cout << "成功导入ply曲面" << endl;

	/*************入口3，左右rgb相机参数*************/
	// 开始写要生成的mtl和obj
	cout << "2222222222开始生成mtl文件2222222222" << endl;
	ofstream mtlFile("DataSave/OrgData/colorfacemesh.mtl");
	mtlFile << "# Wavefront material file" << endl;

	// 只保留顶点颜色材质
	mtlFile << "newmtl VertexColorMaterial" << endl;
	mtlFile << "Ka 1.0 1.0 1.0" << endl;
	mtlFile << "Kd 1.0 1.0 1.0" << endl;
	mtlFile << "Ks 0.0 0.0 0.0" << endl;
	mtlFile << "d 1.0" << endl;  // 不透明
	mtlFile << "Ns 0.0" << endl;
	mtlFile << "illum 0" << endl; // 使用顶点颜色，不需要光照计算
	mtlFile << "###" << endl;
	cout << "成功生成mtl文件（仅包含顶点颜色材质）" << endl;
	mtlFile.close();

	cout << "3333333333开始生成obj文件3333333333" << endl;
	ofstream obj_file("DataSave/OrgData/colorfacemesh.obj");
	obj_file << "# Vertices: " << cloud->points.size() << endl;
	obj_file << "# Faces: " << mesh.polygons.size() << endl;
	obj_file << "# Material information:" << endl;
	obj_file << "mtllib colorfacemesh.mtl" << endl;
	// 移除这行，因为我们不需要纹理贴图
	// obj_file << "usemtl tex_material_0" << endl;
	obj_file << "usemtl VertexColorMaterial" << endl; // 明确指定使用顶点颜色材质
	cout << "材质库信息成功生成" << endl;

	// ============== 点云消缝 - 增强平滑过渡版 ==============
	float y_nose = -60.0f;
	float range = 5.0f;
	//if (range < 0) range = 5.0f;

	PointCloud::Ptr left_original_points(new PointCloud);
	PointCloud::Ptr right_original_points(new PointCloud);
	left_original_points->reserve(cloud->size());
	right_original_points->reserve(cloud->size());


	PointT liftpoint;
	cv::Point2f rgbxy;
	Mat debugTransition = stitchedImage2.clone();
	int texture_coord_count = 0; // 这个计数器在新的逻辑中不再重要，但可以保留

	std::vector<cv::Vec3b> vertex_colors(cloud->size());


	// 写入纹理坐标的代码被移除，但我们仍然需要这个循环来计算顶点颜色
	for (size_t i = 0; i < cloud->points.size(); i++) {
		const auto& point = cloud->points[i];

		if (y_nose - range < point.y && point.y < y_nose + range) {
			// === 过渡区域处理 ===
			// 1. 计算左脸纹理坐标（用于获取颜色）
			PointT liftpoint_left = pointright2lift(point, ICP_r);
			Point2f left_uv = Point2RGBXY(1944, 2592, liftpoint_left, 0);
			float left_u = (left_uv.x - (1944 - height2 - ly)) / col;
			float left_v = 1.00 - (left_uv.y - lx) / row;

			int left_x = static_cast<int>(left_uv.x - (1944 - height2 - ly));
			int left_y = static_cast<int>(left_uv.y - lx);
			circle(debugTransition, Point(left_x, left_y), 5, Scalar(0, 255, 0), -1); // 左投影点：绿色

			// 2. 计算右脸纹理坐标（用于获取颜色）
			Point2f right_uv = Point2RGBXY(1944, 2592, point, 1);
			float right_u = (right_uv.x - (1944 - height2 - ry)) / col;
			float right_v = 1.00 - ((right_uv.y + width2 * rate - rx)) / row;

			int right_x = static_cast<int>(right_uv.x - (1944 - height2 - ry));
			int right_y = static_cast<int>(right_uv.y + width2 * rate - rx);
			circle(debugTransition, Point(right_x, right_y), 5, Scalar(0, 0, 255), -1); // 右投影点：蓝色

			// 3. 计算混合权重
			const float transition_y = -60.0f;
			const float transition_width = 5.0f;
			float dist_to_seam = point.y - transition_y;
			float blend_weight = 0.0f;
			if (dist_to_seam < -transition_width) {
				blend_weight = 1.0f;
			}
			else if (dist_to_seam > transition_width) {
				blend_weight = 0.0f;
			}
			else {
				blend_weight = 1.0f - (dist_to_seam + transition_width) / (2 * transition_width);
			}

			// 4. 获取左右脸原始颜色
			Vec3b left_color = stitchedImage2.at<Vec3b>(
				min(max(left_y, 0), stitchedImage2.rows - 1),
				min(max(left_x, 0), stitchedImage2.cols - 1)
			);
			Vec3b right_color = stitchedImage2.at<Vec3b>(
				min(max(right_y, 0), stitchedImage2.rows - 1),
				min(max(right_x, 0), stitchedImage2.cols - 1)
			);

			// 5. 混合颜色并存储到 vertex_colors
			Vec3b blended_color;
			blended_color[0] = static_cast<uchar>(left_color[0] * blend_weight + right_color[0] * (1 - blend_weight));
			blended_color[1] = static_cast<uchar>(left_color[1] * blend_weight + right_color[1] * (1 - blend_weight));
			blended_color[2] = static_cast<uchar>(left_color[2] * blend_weight + right_color[2] * (1 - blend_weight));
			vertex_colors[i] = blended_color;
		}
		else {
			// === 非过渡区域处理 ===
			if (point.y < y_nose) {
				// 左脸点
				liftpoint = pointright2lift(point, ICP_r);
				rgbxy = Point2RGBXY(1944, 2592, liftpoint, 0);
				int left_x = static_cast<int>(rgbxy.x - (1944 - height2 - ly));
				int left_y = static_cast<int>(rgbxy.y - lx);

				Vec3b left_color = stitchedImage2.at<Vec3b>(
					min(max(left_y, 0), stitchedImage2.rows - 1),
					min(max(left_x, 0), stitchedImage2.cols - 1)
				);
				vertex_colors[i] = left_color;
			}
			else {
				// 右脸点
				rgbxy = Point2RGBXY(1944, 2592, point, 1);
				int right_x = static_cast<int>(rgbxy.x - (1944 - height2 - ry));
				int right_y = static_cast<int>(rgbxy.y + width2 * rate - rx);

				Vec3b right_color = stitchedImage2.at<Vec3b>(
					min(max(right_y, 0), stitchedImage2.rows - 1),
					min(max(right_x, 0), stitchedImage2.cols - 1)
				);
				vertex_colors[i] = right_color;
			}
		}
	}

	// ===== 修改顶点写入部分 =====
	// 写入带RGB的顶点坐标
	for (size_t i = 0; i < cloud->size(); i++) {
		const auto& point = cloud->points[i];
		const auto& color = vertex_colors[i];

		float r = color[2] / 255.0f;
		float g = color[1] / 255.0f;
		float b = color[0] / 255.0f;

		obj_file << "v " << point.x << " " << point.y << " " << point.z << " "
			<< r << " " << g << " " << b << endl;
	}
	cout << "成功写入" << cloud->size() << "个带RGB的顶点" << endl;

	// 点云计算法向量
	PointCloudN::Ptr normals = NormalCalculate(cloud, voxel_size * 2);
	for (const auto& normal : normals->points)
	{
		obj_file << "vn " << normal.normal_x << " " << normal.normal_y << " " << normal.normal_z << endl;
	}cout << "成功生成法向量" << endl;

	// 面生成 - 统一使用顶点颜色格式
	for (const auto& polygon : mesh.polygons)
	{
		obj_file << "f ";
		for (const auto& vertex_index : polygon.vertices) {
			// 使用 v//vn 格式，不包含纹理坐标索引
			obj_file << vertex_index + 1 << "//" << vertex_index + 1 << " ";
		}
		obj_file << endl;
	}
	cout << "成功生成面和索引" << endl;

	/****************出口3，完整的obj****************/
	obj_file.close();
	cout << "成功生成obj文件" << endl;
	cout << "曲面贴图完成！！！" << endl;

	imwrite("DataSave/OrgData/transition_debug.png", debugTransition);

	//// ============= 直方图匹配：替换你的色彩校正（可直接粘贴替换）=============
 //  // 方式A（推荐）：先构建 LUT，再整图应用，同时生成逐点评估颜色
	//std::array<uchar, 256> lutL, lutA, lutB;
	//build_lab_LUTs_from_samples(filteredLeftFeatures, filteredRightFeatures, lutL, lutA, lutB);

	//// 1) 整幅左图做直方图匹配（Lab 三通道，基于样本的目标分布）
	//cv::Mat corrected_left = hist_match_image_lab_from_samples(targetImage3, filteredLeftFeatures, filteredRightFeatures);
	//targetImage3 = corrected_left.clone(); // 覆盖左图，后续流程保持不变

	//// 2) 用同一 LUT 逐点生成“变换后的左侧颜色”，用于你后续的误差/ΔE 评估
	//std::vector<cv::Vec3b> transformedFeatures;
	//transformedFeatures.reserve(filteredLeftFeatures.size());
	//for (size_t i = 0; i < filteredLeftFeatures.size(); ++i) {
	//	transformedFeatures.push_back(
	//		hist_match_point_from_samples(filteredLeftFeatures[i], lutL, lutA, lutB)
	//	);
	//}

	//// 3) （可选）你原有的颜色差/ΔE 评估可直接复用，示例：平均 BGR 欧氏差
	//float transformedMeanDiff = 0.0f;
	//for (size_t i = 0; i < filteredLeftFeatures.size(); ++i) {
	//	const cv::Vec3b& tl = transformedFeatures[i];
	//	const cv::Vec3b& rr = filteredRightFeatures[i];
	//	float d = std::sqrt(
	//		std::pow(float(tl[0]) - rr[0], 2) +
	//		std::pow(float(tl[1]) - rr[1], 2) +
	//		std::pow(float(tl[2]) - rr[2], 2)
	//	);
	//	transformedMeanDiff += d;
	//}
	//transformedMeanDiff /= static_cast<float>(filteredLeftFeatures.size());
	//std::cout << "[HistMatch] 变换后平均颜色差: " << transformedMeanDiff << std::endl;
	//// ============= 直方图匹配：替换区结束 =============

	// ========== Reinhard 色彩传递替换区 ==========
// // —— 替换区：用“整图统计”做 Reinhard 矫正 —— //
	//cv::Vec3f muLeft, stdLeft, muRight, stdRight;

	//// 若你只想用某个 ROI 统计，可传 roi；或传入 mask 仅统计人脸区域
	//// 这里按“整张图”统计：
	//meanStdLab_from_image(targetImage3, muLeft, stdLeft);
	//meanStdLab_from_image(targetImage4, muRight, stdRight);

	//// 把左图整体映射到右图的均值/方差
	//cv::Mat corrected_left = reinhard_transfer_image_BGR(
	//	targetImage3, muLeft, stdLeft, muRight, stdRight
	//);
	//targetImage3 = corrected_left.clone(); // 覆盖左图，后续流程（拼接/验证/ΔE）保持不变

	////// 3) 生成逐点评估用的变换后颜色
	////std::vector<cv::Vec3b> transformedFeatures;
	//transformedFeatures.reserve(filteredLeftFeatures.size());
	//for (size_t i = 0; i < filteredLeftFeatures.size(); ++i) {
	//	transformedFeatures.push_back(
	//		reinhard_transfer_color_BGR(filteredLeftFeatures[i], muLeft, stdLeft, muRight, stdRight)
	//	);
	//}

	//// 4) 变换后均值差
	//transformedMeanDiff = 0.0f;
	////float transformedMeanDiff = 0.0f;
	//for (size_t i = 0; i < filteredLeftFeatures.size(); ++i) {
	//	const cv::Vec3b& transLeft = transformedFeatures[i];
	//	const cv::Vec3b& targetRight = filteredRightFeatures[i];
	//	float diff = std::sqrt(
	//		std::pow(float(transLeft[0]) - targetRight[0], 2) +
	//		std::pow(float(transLeft[1]) - targetRight[1], 2) +
	//		std::pow(float(transLeft[2]) - targetRight[2], 2)
	//	);
	//	transformedMeanDiff += diff;
	//}
	//transformedMeanDiff /= static_cast<float>(filteredLeftFeatures.size());
	//std::cout << "\n[Reinhard] 变换后平均颜色差: " << transformedMeanDiff << std::endl;
	// ========== Reinhard 色彩传递替换区 ==========

	// =============== 打印详细的颜色信息（同时显示两次校正） ===============
	auto euclid_bgr = [](const cv::Vec3b& a, const cv::Vec3b& b) -> float {
		return std::sqrt(
			std::pow(float(a[0]) - b[0], 2) +
			std::pow(float(a[1]) - b[1], 2) +
			std::pow(float(a[2]) - b[2], 2)
		);
		};

	std::cout << "\n特征点颜色对比 (格式: BGR):\n";
	std::cout << "索引 | 原始颜色差 | 变换后(M)差 | 变换后(R)差 | 左原BGR | 左(M)后BGR | 左(R)后BGR | 右目标BGR\n";
	std::cout << "-------------------------------------------------------------------------------------------------------------\n";

	for (size_t i = 0; i < filteredLeftFeatures.size(); ++i)
	{
		int idx = (i < filteredIndices.size() ? filteredIndices[i] : int(i));

		const cv::Vec3b origLeft = filteredLeftFeatures[i];
		const cv::Vec3b left_M = transformedFeatures_M[i];   // 矩阵后
		const cv::Vec3b left_R = transformedFeatures_R[i];   // 矩阵后再 Reinhard
		const cv::Vec3b rightRef = filteredRightFeatures[i];

		// 原始差异：优先用已有 colorDiffs（注意你历史的 +20 偏移），否则现场计算
		float origDiff;
		if (!colorDiffs.empty() && i < colorDiffs.size()) {
			origDiff = colorDiffs[i];
		}
		else if (!colorDiffs.empty() && (filteredIndices[i] - 20) >= 0 && size_t(filteredIndices[i] - 20) < colorDiffs.size()) {
			origDiff = colorDiffs[filteredIndices[i] - 20];
		}
		else {
			origDiff = euclid_bgr(origLeft, rightRef);
		}

		const float diff_M = euclid_bgr(left_M, rightRef);
		const float diff_R = euclid_bgr(left_R, rightRef);

		std::printf("%4d | %12.1f | %12.1f | %12.1f | (%3d,%3d,%3d) | (%3d,%3d,%3d) | (%3d,%3d,%3d) | (%3d,%3d,%3d)\n",
			idx,
			origDiff, diff_M, diff_R,
			origLeft[0], origLeft[1], origLeft[2],
			left_M[0], left_M[1], left_M[2],
			left_R[0], left_R[1], left_R[2],
			rightRef[0], rightRef[1], rightRef[2]
		);
	}

	// =============== ΔE 评估：原始 vs 右、矩阵后 vs 右、Reinhard 后 vs 右 ===============
	{
		const bool size_ok =
			!filteredLeftFeatures.empty() &&
			filteredLeftFeatures.size() == filteredRightFeatures.size() &&
			transformedFeatures_M.size() == filteredRightFeatures.size() &&
			transformedFeatures_R.size() == filteredRightFeatures.size();

		if (!size_ok) {
			std::cout << "[ΔE] 可用样本不足或大小不匹配，跳过 ΔE 评估。\n";
		}
		else {
			const size_t N = filteredLeftFeatures.size();

			std::vector<double> dE76_before, dE76_after_M, dE76_after_R;
			std::vector<double> dE00_before, dE00_after_M, dE00_after_R;
			dE76_before.reserve(N); dE76_after_M.reserve(N); dE76_after_R.reserve(N);
			dE00_before.reserve(N); dE00_after_M.reserve(N); dE00_after_R.reserve(N);

			for (size_t i = 0; i < N; ++i) {
				// BGR->Lab（使用你已有的 BGRu8_to_Lab32f）
				const cv::Vec3f LabL = BGRu8_to_Lab32f(filteredLeftFeatures[i]);   // 原始左
				const cv::Vec3f LabR = BGRu8_to_Lab32f(filteredRightFeatures[i]);  // 右目标
				const cv::Vec3f LabLM = BGRu8_to_Lab32f(transformedFeatures_M[i]);  // 矩阵后
				const cv::Vec3f LabLR = BGRu8_to_Lab32f(transformedFeatures_R[i]);  // Reinhard 后

				// ΔE76 / ΔE2000
				dE76_before.push_back(deltaE76(LabL, LabR));
				dE76_after_M.push_back(deltaE76(LabLM, LabR));
				dE76_after_R.push_back(deltaE76(LabLR, LabR));

				dE00_before.push_back(deltaE2000(LabL, LabR));
				dE00_after_M.push_back(deltaE2000(LabLM, LabR));
				dE00_after_R.push_back(deltaE2000(LabLR, LabR));
			}

			auto print_stats = [](const char* name, const std::vector<double>& v) {
				std::vector<double> tmp = v;
				const double m = mean_val(v);
				const double md = median(std::move(tmp));
				std::vector<double> t90 = v, t95 = v;
				const double p90 = percentile(std::move(t90), 90.0);
				const double p95 = percentile(std::move(t95), 95.0);

				size_t c1 = 0, c2 = 0, c3 = 0;
				for (double x : v) { if (x <= 1.0) ++c1; if (x <= 2.0) ++c2; if (x <= 3.0) ++c3; }
				const double n = static_cast<double>(v.size());

				std::cout << name << " -> "
					<< "mean: " << m
					<< ", median: " << md
					<< ", P90: " << p90
					<< ", P95: " << p95
					<< ", ≤1: " << (c1 / n * 100.0) << "%, "
					<< "≤2: " << (c2 / n * 100.0) << "%, "
					<< "≤3: " << (c3 / n * 100.0) << "%\n";
				};

			auto rel_improve = [](double a, double b) { return (a > 1e-9) ? (a - b) / a * 100.0 : 0.0; };

			std::cout << "\n================ ΔE 指标（Lab） ================\n";
			std::cout << "[ΔE76]  原始(左vs右): ";  print_stats("before", dE76_before);
			std::cout << "[ΔE76]  矩阵后(M):     ";  print_stats("after_M", dE76_after_M);
			std::cout << "[ΔE76]  Reinhard后:   ";  print_stats("after_R", dE76_after_R);

			std::cout << "[ΔE2000]原始(左vs右): ";  print_stats("before", dE00_before);
			std::cout << "[ΔE2000]矩阵后(M):    ";  print_stats("after_M", dE00_after_M);
			std::cout << "[ΔE2000]Reinhard后:  ";  print_stats("after_R", dE00_after_R);

			std::cout << "------------------------------------------------\n";
			std::cout << "ΔE76  mean 改善(↓%)  原始→矩阵："
				<< rel_improve(mean_val(dE76_before), mean_val(dE76_after_M)) << "%, "
				<< "矩阵→Reinhard："
				<< rel_improve(mean_val(dE76_after_M), mean_val(dE76_after_R)) << "%, "
				<< "原始→Reinhard："
				<< rel_improve(mean_val(dE76_before), mean_val(dE76_after_R)) << "%\n";

			std::cout << "ΔE2000 mean 改善(↓%) 原始→矩阵："
				<< rel_improve(mean_val(dE00_before), mean_val(dE00_after_M)) << "%, "
				<< "矩阵→Reinhard："
				<< rel_improve(mean_val(dE00_after_M), mean_val(dE00_after_R)) << "%, "
				<< "原始→Reinhard："
				<< rel_improve(mean_val(dE00_before), mean_val(dE00_after_R)) << "%\n";
			std::cout << "================================================\n";
		}
	}


	//// 打印详细的颜色信息（仅显示筛选后的点）
	//std::cout << "\n特征点颜色对比 (格式: BGR):" << std::endl;
	//std::cout << "索引 | 原始颜色差异 | 变换后颜色差异 | 左脸原始BGR | 左脸变换后BGR | 右脸目标BGR" << std::endl;
	//std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
	//for (size_t i = 0; i < filteredLeftFeatures.size(); i++)
	//{
	//	int idx = filteredIndices[i];
	//	cv::Vec3b origLeft = filteredLeftFeatures[i];
	//	cv::Vec3b transLeft = transformedFeatures[i];
	//	cv::Vec3b targetRight = filteredRightFeatures[i];
	//	float origDiff = colorDiffs[filteredIndices[i] - 20];

	//	float transDiff = std::sqrt(
	//		std::pow(transLeft[0] - targetRight[0], 2) +
	//		std::pow(transLeft[1] - targetRight[1], 2) +
	//		std::pow(transLeft[2] - targetRight[2], 2)
	//	);

	//	std::printf("%2d | %10.1f | %10.1f | (%3d,%3d,%3d) | (%3d,%3d,%3d) | (%3d,%3d,%3d)\n",
	//		idx,
	//		origDiff,
	//		transDiff,
	//		origLeft[0], origLeft[1], origLeft[2],
	//		transLeft[0], transLeft[1], transLeft[2],
	//		targetRight[0], targetRight[1], targetRight[2]);
	//}

	//// ===== ΔE 评估（新增，不改动既有流程）=====
	//{
	//	if (filteredLeftFeatures.empty() || filteredRightFeatures.empty() ||
	//		filteredLeftFeatures.size() != filteredRightFeatures.size() ||
	//		transformedFeatures.size() != filteredRightFeatures.size()) {
	//		std::cout << "[ΔE] 可用样本不足或大小不匹配，跳过 ΔE 评估。\n";
	//	}
	//	else {
	//		const size_t N = filteredLeftFeatures.size();
	//		std::vector<double> dE76_before, dE76_after, dE00_before, dE00_after;
	//		dE76_before.reserve(N); dE76_after.reserve(N);
	//		dE00_before.reserve(N); dE00_after.reserve(N);

	//		for (size_t i = 0; i < N; ++i) {
	//			// BGR->Lab
	//			cv::Vec3f LabL = BGR2Lab32f(filteredLeftFeatures[i]);
	//			cv::Vec3f LabR = BGR2Lab32f(filteredRightFeatures[i]);
	//			cv::Vec3f LabLt = BGR2Lab32f(transformedFeatures[i]);

	//			// ΔE76 & ΔE2000 : Before (原始左 vs 右) / After (变换后左 vs 右)
	//			dE76_before.push_back(deltaE76(LabL, LabR));
	//			dE76_after.push_back(deltaE76(LabLt, LabR));
	//			dE00_before.push_back(deltaE2000(LabL, LabR));
	//			dE00_after.push_back(deltaE2000(LabLt, LabR));
	//		}

	//		auto print_stats = [](const char* name, const std::vector<double>& v) {
	//			std::vector<double> tmp = v;
	//			const double m = mean_val(v);
	//			const double md = median(std::move(tmp));
	//			std::vector<double> t90 = v, t95 = v;
	//			const double p90 = percentile(std::move(t90), 90.0);
	//			const double p95 = percentile(std::move(t95), 95.0);

	//			size_t c1 = 0, c2 = 0, c3 = 0;
	//			for (double x : v) { if (x <= 1.0) ++c1; if (x <= 2.0) ++c2; if (x <= 3.0) ++c3; }
	//			const double n = static_cast<double>(v.size());

	//			std::cout << name << " -> "
	//				<< "mean: " << m
	//				<< ", median: " << md
	//				<< ", P90: " << p90
	//				<< ", P95: " << p95
	//				<< ", ≤1: " << (c1 / n * 100.0) << "%, "
	//				<< "≤2: " << (c2 / n * 100.0) << "%, "
	//				<< "≤3: " << (c3 / n * 100.0) << "%\n";
	//			};

	//		std::cout << "\n================ ΔE 指标（Lab） ================\n";
	//		std::cout << "[ΔE76]  原始(左vs右): ";  print_stats("before", dE76_before);
	//		std::cout << "[ΔE76]  校正(左'vs右): "; print_stats("after ", dE76_after);
	//		std::cout << "[ΔE2000]原始(左vs右): ";  print_stats("before", dE00_before);
	//		std::cout << "[ΔE2000]校正(左'vs右): "; print_stats("after ", dE00_after);

	//		auto rel_improve = [](double a, double b) { return (a > 1e-9) ? (a - b) / a * 100.0 : 0.0; };
	//		std::cout << "------------------------------------------------\n";
	//		std::cout << "ΔE76  mean 改善(↓%)："
	//			<< rel_improve(mean_val(dE76_before), mean_val(dE76_after)) << "%\n";
	//		std::cout << "ΔE2000 mean 改善(↓%)："
	//			<< rel_improve(mean_val(dE00_before), mean_val(dE00_after)) << "%\n";
	//		std::cout << "================================================\n";
	//	}
	//}

	// 保存结果
	cv::imwrite("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/corrected_left_cpp.png", targetImage3);


	// 定义变量保存拼接后的图像
	Mat stitchedImage2(height2, width2 * rate * 2, CV_8UC3);
	// 左右矩形各一半（各占 rate）
	Mat leftROI2 = stitchedImage2(Rect(0, 0, width2 * rate, height2));
	Mat rightROI2 = stitchedImage2(Rect(width2 * rate, 0, width2 * rate, height2));

	targetImage3(Rect(0, 0, width2 * rate, height2)).copyTo(leftROI2);
	targetImage4(Rect(width2 * (1.0 - rate), 0, width2 * rate, height2)).copyTo(rightROI2);
	cout << "rgb图片拼接完成" << endl;

	// 继续后续的旋转和保存操作...
	stitchedImage2 = RGB90right(stitchedImage2);
	cv::imwrite("C:/Users/tester/Desktop/FaceStitche_QT/FaceStitche/DataSave/OrgData/stitchedImage2.png", stitchedImage2);

	cout << "开始处理曲面" << endl;

	/****************入口2，曲面mesh****************/
	cout << "成功导入ply曲面" << endl;

	/*************入口3，左右rgb相机参数*************/
	// 开始写要生成的mtl和obj
	cout << "2222222222开始生成mtl文件2222222222" << endl;
	ofstream mtlFile("DataSave/OrgData/colorfacemesh.mtl");
	mtlFile << "# Wavefront material file" << endl;

	// 只保留顶点颜色材质
	mtlFile << "newmtl VertexColorMaterial" << endl;
	mtlFile << "Ka 1.0 1.0 1.0" << endl;
	mtlFile << "Kd 1.0 1.0 1.0" << endl;
	mtlFile << "Ks 0.0 0.0 0.0" << endl;
	mtlFile << "d 1.0" << endl;  // 不透明
	mtlFile << "Ns 0.0" << endl;
	mtlFile << "illum 0" << endl; // 使用顶点颜色，不需要光照计算
	mtlFile << "###" << endl;
	cout << "成功生成mtl文件（仅包含顶点颜色材质）" << endl;
	mtlFile.close();

	cout << "3333333333开始生成obj文件3333333333" << endl;
	ofstream obj_file("DataSave/OrgData/colorfacemesh.obj");
	obj_file << "# Vertices: " << cloud->points.size() << endl;
	obj_file << "# Faces: " << mesh.polygons.size() << endl;
	obj_file << "# Material information:" << endl;
	obj_file << "mtllib colorfacemesh.mtl" << endl;
	// 移除这行，因为我们不需要纹理贴图
	// obj_file << "usemtl tex_material_0" << endl;
	obj_file << "usemtl VertexColorMaterial" << endl; // 明确指定使用顶点颜色材质
	cout << "材质库信息成功生成" << endl;

	// ============== 点云消缝 - 增强平滑过渡版 ==============
	float y_nose = -60.0f;
	float range = 5.0f;
	//if (range < 0) range = 5.0f;

	PointCloud::Ptr left_original_points(new PointCloud);
	PointCloud::Ptr right_original_points(new PointCloud);
	left_original_points->reserve(cloud->size());
	right_original_points->reserve(cloud->size());


	PointT liftpoint;
	cv::Point2f rgbxy;
	Mat debugTransition = stitchedImage2.clone();
	int texture_coord_count = 0; // 这个计数器在新的逻辑中不再重要，但可以保留

	std::vector<cv::Vec3b> vertex_colors(cloud->size());


	// 写入纹理坐标的代码被移除，但我们仍然需要这个循环来计算顶点颜色
	for (size_t i = 0; i < cloud->points.size(); i++) {
		const auto& point = cloud->points[i];

		if (y_nose - range < point.y && point.y < y_nose + range) {
			// === 过渡区域处理 ===
			// 1. 计算左脸纹理坐标（用于获取颜色）
			PointT liftpoint_left = pointright2lift(point, ICP_r);
			Point2f left_uv = Point2RGBXY(1944, 2592, liftpoint_left, 0);
			float left_u = (left_uv.x - (1944 - height2 - ly)) / col;
			float left_v = 1.00 - (left_uv.y - lx) / row;

			int left_x = static_cast<int>(left_uv.x - (1944 - height2 - ly));
			int left_y = static_cast<int>(left_uv.y - lx);
			circle(debugTransition, Point(left_x, left_y), 5, Scalar(0, 255, 0), -1); // 左投影点：绿色

			// 2. 计算右脸纹理坐标（用于获取颜色）
			Point2f right_uv = Point2RGBXY(1944, 2592, point, 1);
			float right_u = (right_uv.x - (1944 - height2 - ry)) / col;
			float right_v = 1.00 - ((right_uv.y + width2 * rate - rx)) / row;

			int right_x = static_cast<int>(right_uv.x - (1944 - height2 - ry));
			int right_y = static_cast<int>(right_uv.y + width2 * rate - rx);
			circle(debugTransition, Point(right_x, right_y), 5, Scalar(0, 0, 255), -1); // 右投影点：蓝色

			// 3. 计算混合权重
			const float transition_y = -60.0f;
			const float transition_width = 5.0f;
			float dist_to_seam = point.y - transition_y;
			float blend_weight = 0.0f;
			if (dist_to_seam < -transition_width) {
				blend_weight = 1.0f;
			}
			else if (dist_to_seam > transition_width) {
				blend_weight = 0.0f;
			}
			else {
				blend_weight = 1.0f - (dist_to_seam + transition_width) / (2 * transition_width);
			}

			// 4. 获取左右脸原始颜色
			Vec3b left_color = stitchedImage2.at<Vec3b>(
				min(max(left_y, 0), stitchedImage2.rows - 1),
				min(max(left_x, 0), stitchedImage2.cols - 1)
			);
			Vec3b right_color = stitchedImage2.at<Vec3b>(
				min(max(right_y, 0), stitchedImage2.rows - 1),
				min(max(right_x, 0), stitchedImage2.cols - 1)
			);

			// 5. 混合颜色并存储到 vertex_colors
			Vec3b blended_color;
			blended_color[0] = static_cast<uchar>(left_color[0] * blend_weight + right_color[0] * (1 - blend_weight));
			blended_color[1] = static_cast<uchar>(left_color[1] * blend_weight + right_color[1] * (1 - blend_weight));
			blended_color[2] = static_cast<uchar>(left_color[2] * blend_weight + right_color[2] * (1 - blend_weight));
			vertex_colors[i] = blended_color;
		}
		else {
			// === 非过渡区域处理 ===
			if (point.y < y_nose) {
				// 左脸点
				liftpoint = pointright2lift(point, ICP_r);
				rgbxy = Point2RGBXY(1944, 2592, liftpoint, 0);
				int left_x = static_cast<int>(rgbxy.x - (1944 - height2 - ly));
				int left_y = static_cast<int>(rgbxy.y - lx);

				Vec3b left_color = stitchedImage2.at<Vec3b>(
					min(max(left_y, 0), stitchedImage2.rows - 1),
					min(max(left_x, 0), stitchedImage2.cols - 1)
				);
				vertex_colors[i] = left_color;
			}
			else {
				// 右脸点
				rgbxy = Point2RGBXY(1944, 2592, point, 1);
				int right_x = static_cast<int>(rgbxy.x - (1944 - height2 - ry));
				int right_y = static_cast<int>(rgbxy.y + width2 * rate - rx);

				Vec3b right_color = stitchedImage2.at<Vec3b>(
					min(max(right_y, 0), stitchedImage2.rows - 1),
					min(max(right_x, 0), stitchedImage2.cols - 1)
				);
				vertex_colors[i] = right_color;
			}
		}
	}

	// ===== 修改顶点写入部分 =====
	// 写入带RGB的顶点坐标
	for (size_t i = 0; i < cloud->size(); i++) {
		const auto& point = cloud->points[i];
		const auto& color = vertex_colors[i];

		float r = color[2] / 255.0f;
		float g = color[1] / 255.0f;
		float b = color[0] / 255.0f;

		obj_file << "v " << point.x << " " << point.y << " " << point.z << " "
			<< r << " " << g << " " << b << endl;
	}
	cout << "成功写入" << cloud->size() << "个带RGB的顶点" << endl;

	// 点云计算法向量
	PointCloudN::Ptr normals = NormalCalculate(cloud, voxel_size * 2);
	for (const auto& normal : normals->points)
	{
		obj_file << "vn " << normal.normal_x << " " << normal.normal_y << " " << normal.normal_z << endl;
	}cout << "成功生成法向量" << endl;

	// 面生成 - 统一使用顶点颜色格式
	for (const auto& polygon : mesh.polygons)
	{
		obj_file << "f ";
		for (const auto& vertex_index : polygon.vertices) {
			// 使用 v//vn 格式，不包含纹理坐标索引
			obj_file << vertex_index + 1 << "//" << vertex_index + 1 << " ";
		}
		obj_file << endl;
	}
	cout << "成功生成面和索引" << endl;

	/****************出口3，完整的obj****************/
	obj_file.close();
	cout << "成功生成obj文件" << endl;
	cout << "曲面贴图完成！！！" << endl;

	imwrite("DataSave/OrgData/transition_debug.png", debugTransition);