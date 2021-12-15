#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <vector>
#include <chrono>
#include <string>
#include <vector>
//using namespace cv;
//using namespace std;


torch::Tensor process( cv::Mat& image, torch::Device device, int img_W, int img_H)
{
    vector <float> mean_ = {0.5, 0.5, 0.5};
    vector <float> std_ = {0.5, 0.5, 0.5};
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);// bgr -> rgb
    cv::Mat img_float;
//    image.convertTo(img_float, CV_32F, 1.0 / 255);//归一化到[0,1]区间,
    cv::resize(image, img_float, cv::Size(img_W, img_H));

    std::vector<int64_t> dims = {1, img_H, img_W, 3};
    torch::Tensor img_var = torch::from_blob(img_float.data, dims, torch::kByte).to(device);//将图像转化成张量
    img_var = img_var.permute({0,3,1,2});//将张量的参数顺序转化为 torch输入的格式 NCHW
    img_var = img_var.toType(torch::kFloat);
    img_var = img_var.div(255);

    for (int i = 0; i < 3; i++) {
        img_var[0][i] = img_var[0][i].sub_(mean_[i]).div_(std_[i]);
    }
    img_var = torch::autograd::make_variable(img_var, false);
    return img_var;
}

int main(){


    string img_path = "../person.jpg";
    string model_path = "../model.pt";
    int img_W = 640;
    int img_H = 480;

    //加载模型到CPU
    torch::DeviceType device_type;
	device_type = torch::kCPU;
	torch::Device device(device_type);
	torch::jit::script::Module module = torch::jit::load(model_path);//
	module.to(torch::kCPU);
    module.eval();
    std::cout<<"Model loading complete")<<std::endl;

     // 读取图片
    cv::Mat image = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);
    if (image.empty())
        std::cout<<"Can not load image!")<<std::endl;

    torch::Tensor img_var=process(image, device, img_W, img_H);

    //forward
    torch::Tensor result = module.forward({img_var}).toTensor();
    result = result.squeeze();//删除1的维度
    result = result.mul(255).to(torch::kU8) ;
    cv::Mat pts_mat(cv::Size(img_W, img_H), CV_8U, result.data_ptr());
    cv::imwrite("../matting_person.png", pts_mat);
    std::cout<<"matting complete!")<<std::endl;

}