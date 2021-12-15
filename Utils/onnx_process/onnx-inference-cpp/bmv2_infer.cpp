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

static constexpr const int width_ = 1280;
static constexpr const int height_ = 720;
static constexpr const int channel = 3;

std::array<float, 1 * width_ * height_ * channel> input_src_{};
std::array<float, 1 * width_ * height_ * channel> input_bgr_{};
std::array<int64_t, 4> input_shape_{ 1, channel, height_, width_ };


int main()
{
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
	Ort::SessionOptions session_option;
	session_option.SetIntraOpNumThreads(4);

#ifdef _WIN32
	const wchar_t* model_path = L"D://ONNX_test/bmv2_refine.onnx";
#else
	const char* model_path = "D://ONNX_test/bmv2_refine.onnx";
#endif

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_option);
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	size_t num_input_nodes = session.GetInputCount();
	vector<const char*> input_node_names = { "src", "bgr" };
	vector<const char*> output_node_names = { "pha", "fgr" };

	const int row = height_;
	const int col = width_;

	Mat src = imread("D://ONNX_test/src.png");
	Mat dst(row, col, CV_8UC3);
	float* src_values = input_src_.data();
	fill(input_src_.begin(), input_src_.end(), 0.f);
	resize(src, dst, Size(col, row), INTER_AREA);
	cvtColor(dst, dst, CV_BGR2RGB);
	Ort::Value input_src_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_src_.data(), input_src_.size(), input_shape_.data(), input_shape_.size());
	// inline Value Value::CreateTensor(const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len)
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				src_values[c * row * col + i * col + j] = (dst.ptr<uchar>(i)[j * 3 + c]) / 255.0;
			}
		}
	}

	Mat bgr = imread("D://ONNX_test/bg.png");
	float* bgr_values = input_bgr_.data();
	fill(input_bgr_.begin(), input_bgr_.end(), 0.f);
	resize(bgr, dst, Size(col, row), INTER_AREA);
	cvtColor(dst, dst, CV_BGR2RGB);
	Ort::Value input_bgr_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_bgr_.data(), input_bgr_.size(), input_shape_.data(), input_shape_.size());
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				bgr_values[c * row * col + i * col + j] = (dst.ptr<uchar>(i)[j * 3 + c]) / 255.0;
			}
		}
	}

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_src_tensor));
	ort_inputs.push_back(std::move(input_bgr_tensor));
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);
	float* alpha = output_tensors[0].GetTensorMutableData<float>();
	float* fgr = output_tensors[1].GetTensorMutableData<float>();

	Mat saveimg(row, col, CV_8UC3);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				saveimg.ptr<uchar>(i)[j * 3 + c] = alpha[i * col + j] * 255 * fgr[c * row * col + i * col + j];
			}
		}
	}
	cvtColor(saveimg, saveimg, CV_BGR2RGB);
	resize(saveimg, saveimg, Size(src.cols, src.rows), INTER_AREA);
	imwrite("D://ONNX_test/test_final.jpg", saveimg);

	system("pause");
	return 0;
}