#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
using namespace cv;
using namespace std;

static constexpr const int width_ = 1280;
static constexpr const int height_ = 720;
static constexpr const int channel = 3;


std::array<float, 1 * width_ * height_ * channel> input_src_{};
std::array<float, 1 * 1 * 1 * 1> r1i{};
std::array<float, 1 * 1 * 1 * 1> r2i{};
std::array<float, 1 * 1 * 1 * 1> r3i{};
std::array<float, 1 * 1 * 1 * 1> r4i{};
std::array<float, 1> downsample_ratio{};

std::array<int64_t, 4> input_shape_{ 1, channel, height_, width_ };
std::array<int64_t, 4> r_shape_{ 1, 1, 1, 1 };
std::array<int64_t, 1> downsample_ratio_shape_{1};
int main()
{
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(4);

#ifdef _WIN32
    const wchar_t* model_path = L"/Users/cz/PycharmProjects/rvm/rvm_mobilenetv3_fp32.onnx";
#else
    const char* model_path = "/Users/cz/PycharmProjects/rvm/rvm_mobilenetv3_fp32.onnx";
#endif

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_option);
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    size_t num_input_nodes = session.GetInputCount();
    vector<const char*> input_node_names = { "src", "r1i", "r2i","r3i","r4i","downsample_ratio" };
    vector<const char*> output_node_names = { "fgr", "pha", "r1o", "r2o", "r3o", "r4o" };

    const int row = height_;
    const int col = width_;

    Mat src = imread("/Users/cz/PycharmProjects/rvm/1.jpg"); // 源图
    Mat src_dst(row, col, CV_8UC3);
    float* src_values = input_src_.data();//获取input_src_地址到src_values
    fill(input_src_.begin(), input_src_.end(), 0.f);
    resize(src, src_dst, Size(col, row), INTER_AREA);
    cvtColor(src_dst, src_dst, CV_BGR2RGB); // 处理后的原图为dst
    Ort::Value input_src_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_src_.data(), input_src_.size(), input_shape_.data(), input_shape_.size());
 
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                src_values[c * row * col + i * col + j] = (src_dst.ptr<uchar>(i)[j * 3 + c]) / 255.0;
            }
        }
    }
    fill(r1i.begin(), r1i.end(), 0.f);
    Ort::Value r1i_tensor = Ort::Value::CreateTensor<float>(allocator_info, r1i.data(), r1i.size(), r_shape_.data(), r_shape_.size());
    
    fill(r2i.begin(), r2i.end(), 0.f);
    Ort::Value r2i_tensor = Ort::Value::CreateTensor<float>(allocator_info, r2i.data(), r2i.size(), r_shape_.data(), r_shape_.size());
    
    
    fill(r3i.begin(), r3i.end(), 0.f);
    Ort::Value r3i_tensor = Ort::Value::CreateTensor<float>(allocator_info, r3i.data(), r3i.size(), r_shape_.data(), r_shape_.size());
    
    
    fill(r4i.begin(), r4i.end(), 0.f);
    Ort::Value r4i_tensor = Ort::Value::CreateTensor<float>(allocator_info, r4i.data(), r4i.size(), r_shape_.data(), r_shape_.size());
    
    downsample_ratio[0] = 0.25;
    Ort::Value downsample_ratio_tensor = Ort::Value::CreateTensor<float>(allocator_info, downsample_ratio.data(), downsample_ratio.size(), downsample_ratio_shape_.data(), downsample_ratio_shape_.size());
    
    
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_src_tensor));
    ort_inputs.push_back(std::move(r1i_tensor));
    ort_inputs.push_back(std::move(r2i_tensor));
    ort_inputs.push_back(std::move(r3i_tensor));
    ort_inputs.push_back(std::move(r4i_tensor));
    ort_inputs.push_back(std::move(downsample_ratio_tensor));
    
    //ort_inputs.push_back(std::move(r1i));
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);
    
    float* fgr = output_tensors[0].GetTensorMutableData<float>();
    float* pha = output_tensors[1].GetTensorMutableData<float>();

    Mat alpha(row, col, CV_8UC3);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                alpha.ptr<uchar>(i)[j * 3 + c] = pha[i * col + j] * 255 ;
            }
        }
    }
    cvtColor(alpha, alpha, CV_BGR2RGB);
    resize(alpha, alpha, Size(src.cols, src.rows), INTER_AREA);
    imwrite("/Users/cz/PycharmProjects/rvm/alpha.png", alpha);
    
    Mat com(row, col, CV_8UC3);
    Mat bgr = imread("/Users/cz/PycharmProjects/rvm/bgr.png"); // 源图
    Mat bgr_dst(row, col, CV_8UC3);
    resize(bgr, bgr_dst, Size(col, row), INTER_AREA);
    cvtColor(bgr_dst, bgr_dst, CV_BGR2RGB); // 处理后的原图为dst
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                com.ptr<uchar>(i)[j * 3 + c] = pha[i * col + j] * src_dst.ptr<uchar>(i)[j * 3 + c] + (1 - pha[i * col + j]) * bgr_dst.ptr<uchar>(i)[j * 3 + c] ;
            }
        }
    }
    resize(com, com, Size(src.cols, src.rows), INTER_AREA);
    cvtColor(com, com, CV_BGR2RGB); // 处理后的原图为dst
    imwrite("/Users/cz/PycharmProjects/rvm/com.png", com);
    //system("pause");
    return 0;
}
