#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;

static constexpr const int width_ = 192;
static constexpr const int height_ = 192;
static constexpr const int channel = 3;

std::array<float, 1 * width_ * height_ * channel> input_src_{};
std::array<float, 1 * width_ * height_ * channel> input_bgr_{};
std::array<int64_t, 4> input_shape_{ 1, channel, height_, width_ };


int main()
{
	Ort::Env env{ OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test" };
	Ort::SessionOptions session_option;
	session_option.SetIntraOpNumThreads(4);

#ifdef _WIN32
	const wchar_t* model_path = L"D://ONNX_test/fcn_seg.onnx";
#else
	const char* model_path = "D://ONNX_test/fcn_seg.onnx";
#endif

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_option);
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	size_t num_input_nodes = session.GetInputCount();
	vector<const char*> input_node_names = { "x" };
	vector<const char*> output_node_names = { "save_infer_model/scale_0.tmp_1" };

	const int row = height_;
	const int col = width_;

	Mat src = imread("D://ONNX_test/src.png");
	Mat dst(row, col, CV_8UC3);
	float* src_values = input_src_.data();
	fill(input_src_.begin(), input_src_.end(), 0.f);
	resize(src, dst, Size(col, row), INTER_AREA);
	Ort::Value input_src_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_src_.data(), input_src_.size(), input_shape_.data(), input_shape_.size());
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				src_values[c * row * col + i * col + j] = (dst.ptr<uchar>(i)[j * 3 + c]) / 255.0;
			}
		}
	}

	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_src_tensor, 1, output_node_names.data(), 1);
	float* alpha = output_tensors[0].GetTensorMutableData<float>();
	Mat scoremap(row, col, CV_8UC1);
	for (int i = 0; i < row * col; i++) {
		scoremap.ptr<uchar>(i / col)[i % col] = (1 - alpha[i]) * 255;
	}
	resize(scoremap, scoremap, Size(src.cols, src.rows), INTER_AREA);

	Mat saveimg(src.rows, src.cols, CV_8UC3);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				saveimg.ptr<uchar>(i)[j * 3 + c] = scoremap.ptr<uchar>(i)[j] / 255.0 * src.ptr<uchar>(i)[j * 3 + c];
			}
		}
	}
	imwrite("D://ONNX_test/test_final.jpg", saveimg);

	system("pause");
	return 0;
}