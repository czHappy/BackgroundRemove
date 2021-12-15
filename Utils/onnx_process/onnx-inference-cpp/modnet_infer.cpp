#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;

static constexpr const int width_ = 512;
static constexpr const int height_ = 512;
static constexpr const int channel = 3;

std::array<float, 1 * width_ * height_ * channel> input_image_{};
std::array<float, 1 * 1 * height_ * width_> results_{};
int result_[height_ * width_]{ 0 };
Mat outputimage(height_, width_, CV_8UC1);

Ort::Value input_tensor_{ nullptr };
std::array<int64_t, 4> input_shape_{ 1,channel, height_, width_ };

Ort::Value output_tensor_{ nullptr };
std::array<int64_t, 4> output_shape_{ 1,1,height_, width_ };


int main()
{
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
	Ort::SessionOptions session_option;
	session_option.SetIntraOpNumThreads(1);

#ifdef _WIN32
	const wchar_t* model_path = L"D://ONNX_test/modnet_photo.onnx";
#else
	const char* model_path = "D://ONNX_test/modnet_photo.onnx";
#endif

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_option);

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	Ort::Value output_tensor = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
	const char* input_names[] = { "input" };
	const char* output_names[] = { "output" };

	Mat img = imread("D://ONNX_test/demo.jpg");
	const int row = height_;
	const int col = width_;
	Mat dst(row, col, CV_8UC3);
	resize(img, dst, Size(col, row));
	cvtColor(dst, dst, CV_BGR2RGB);

	float* output = input_image_.data();
	fill(input_image_.begin(), input_image_.end(), 0.f);
	for (int c = 0; c < 3; c++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				output[c * row * col + i * col + j] = (dst.ptr<uchar>(i)[j * 3 + c] - 127.5) / 127.5;
			}
		}
	}

	double timeStart = (double)getTickCount();
	session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "running time :" << nTime << "sec\n" << endl;

	for (int i = 0; i < height_ * width_; i++) {
		outputimage.ptr<uchar>(i / width_)[i % width_] = results_[i] * 255;
	}
	resize(outputimage, outputimage, Size(img.cols, img.rows));
	imwrite("D://ONNX_test/demo_out.jpg", outputimage);

	system("pause");
	return 0;
}