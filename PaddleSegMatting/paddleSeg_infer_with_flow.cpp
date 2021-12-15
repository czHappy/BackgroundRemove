#include <iostream>
#include <conio.h>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <onnxruntime_cxx_api.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xindex_view.hpp>

#ifdef _WIN32
const wchar_t* model_path = L"D://ONNX_test/fcn_seg.onnx";
#else
const char* model_path = "D://ONNX_test/fcn_seg.onnx";
#endif

using namespace std;
using namespace cv;

static void help()
{
    printf("Usage: dis_optflow <video_file>\n");
}

static constexpr const int width_ = 192;
static constexpr const int height_ = 192;
static constexpr const int channel = 3;

std::array<float, 1 * width_ * height_ * channel> input_src_{};
std::array<float, 1 * width_ * height_ * channel> input_bgr_{};
std::array<int64_t, 4> input_shape_{ 1, channel, height_, width_ };


int main(int argc, char** argv)
{
   /* xt::xarray<int> testt = xt::zeros<int>({ 2, 2 });
    xt::xarray<bool> tb = xt::zeros<bool>({ 2, 2 });
    xt::xarray<int> testt2 = testt;
    testt(1, 1) = 1;
    tb(1, 1) = 1;
    cout << testt << endl;
    cout << testt2 << endl;
    cout << (testt >= .1) << endl;
    xt::xarray<bool> tc = (testt >= .1) * tb;
    tc(0, 0) = 1;*/
    //xt::xarray<bool> tb = xt::zeros<bool>({ 2, 2 });
    //xt::xarray<bool> tc = xt::ones<bool>({ 2, 2 });
    //xt::xarray<bool> ta = xt::zeros<bool>({ 2, 2 });
    //cout << (tc) << endl;
    //cout << (tb) << endl;
    //cout << (tc + tb + tc) << endl;
    //return 0;

    VideoCapture cap;
    if (argc < 2)
    {
        help();
        exit(1);
    }
    cap.open(argv[1]);
    if (!cap.isOpened())
    {
        printf("ERROR: Cannot open file %s\n", argv[1]);
        return -1;
    }
    const int row = height_;
    const int col = width_;
    Mat prev_gray(row, col, CV_8UC1); //前一帧灰度图
    Mat cur_gray(row, col, CV_8UC1);//当前帧灰度图
    Mat frame;//当前帧
    Mat flow_f;//前向光流
    Mat flow_b;//后向光流
    Ptr<DenseOpticalFlow> algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_ULTRAFAST);
    xt::xarray<float> prev_cfd = xt::ones<float>({ row, col, 2 });//
    Ort::Env env{ OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test" };
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(4);

    int id = 0;
    double sum = 0.0;
    int ch;

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_option);
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    size_t num_input_nodes = session.GetInputCount();
    vector<const char*> input_node_names = { "x" };
    vector<const char*> output_node_names = { "save_infer_model/scale_0.tmp_1" };

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        if (_kbhit()) {         //如果有按键按下，则_kbhit()函数返回真
            ch = _getch();      //使用_getch()函数获取按下的键值
            if (ch == 27) { break; }//ESC退出
        }
        id++;
        Mat dst(row, col, CV_8UC3);
        resize(frame, dst, Size(col, row), INTER_AREA);
        cvtColor(dst, cur_gray, COLOR_BGR2GRAY);
        //cout << "cur_gray: "<<cur_gray << endl;
        float* src_values = input_src_.data();
        fill(input_src_.begin(), input_src_.end(), 0.f);
        Ort::Value input_src_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_src_.data(), input_src_.size(), input_shape_.data(), input_shape_.size());
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    src_values[c * row * col + i * col + j] = (dst.ptr<uchar>(i)[j * 3 + c]) / 255.0;
                }
            }
        }
        double timeStart = (double)getTickCount();
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_src_tensor, 1, output_node_names.data(), 1);
        double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
        sum += nTime;
        float* alpha = output_tensors[0].GetTensorMutableData<float>();
        Mat scoremap(row, col, CV_8UC1);
        //xt::xarray<float, xt::layout_type::row_major> cur_cfd({ row, col });
        xt::xarray<float> cur_cfd = xt::ones<float>({ row, col });
        xt::xarray<float> scoremap1 = xt::ones<float>({ row, col });
        for (int i = 0; i < row * col; i++) {
            scoremap.ptr<uchar>(i / col)[i % col] = (1 - alpha[i]) * 255;
            cur_cfd(i / col, i % col) = (1 - alpha[i]) * 255;
            scoremap1(i / col, i % col) = (1 - alpha[i]) * 255;
        }

        if (id < 2) {
            //std::swap(prev_gray, cur_gray);
           // prev_gray = cur_gray;
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    prev_gray.ptr<uchar>(i)[j] = cur_gray.ptr<uchar>(i)[j];
                    prev_cfd(i, j) = cur_cfd(i, j);
                }
            }
         // algorithm->setFinestScale(3);
            //cout << "prev_gray id < 2: " << prev_gray << endl;
            continue;
        }
        else {
            cout << "coming..." << endl;

            cout << cur_gray.size() << endl;
        }


        //*********************光流处理****************************
        int w = col, h = row;
        int check_thres = 8;
        xt::xarray<float> track_cfd = xt::zeros<float>({ row, col });
        xt::xarray<bool> is_track = xt::zeros<bool>({ row, col });

        xt::xarray<int> flow_fw = xt::zeros<int>({ row, col, 2 });
        xt::xarray<int> flow_bw = xt::zeros<int>({ row, col, 2 });

        cout << "before calc" << endl;
        cout << "-------------------------------------" << endl;
        algorithm->calc(prev_gray, cur_gray, flow_f);
        // cout << flow_f << endl;
        algorithm->calc(cur_gray, prev_gray, flow_b);
        //cout << flow_b << endl;
        for (int c = 0; c < 2; c++) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    //cout << flow_ff[c].ptr<int>(i)[j] << " here : " << endl;
                    flow_fw(i, j, c) = (int)flow_f.ptr<float>(i)[j * 2 + c];
                    flow_bw(i, j, c) = (int)flow_b.ptr<float>(i)[j * 2 + c];
                    //if(flow_bw(i, j, c)!=0) cout<< " wow ";
                }
            }
        }

        xt::xarray <int> y_list = xt::arange(0, h);
        xt::xarray <int> x_list = xt::arange(0, w);
        auto xy = xt::meshgrid(y_list, x_list);
        xt::xarray <int> yv = std::get<0>(xy);
        xt::xarray <int> xv = std::get<1>(xy);
        yv = xt::transpose(yv, { 1, 0 });
        xv = xt::transpose(xv, { 1, 0 });
        xt::xarray <int> cur_x = xv + xt::view(flow_fw, xt::all(), xt::all(), 0);
        xt::xarray <int> cur_y = yv + xt::view(flow_fw, xt::all(), xt::all(), 1);
        xt::xarray <bool> not_track = (cur_x < 0) | (cur_x >= w) | (cur_y < 0) | (cur_y >= h);
        //for (int i = 0; i < row; i++) {
        //    for (int j = 0; j < col; j++) {
        //        //cout << flow_ff[c].ptr<int>(i)[j] << " here : " << endl;
        //        if (not_track(i, j) == true) cout << "*****************8888" << endl;

        //    }
        //}
        //cout << xt::view(flow_fw, xt::all(), xt::all(), 0) << endl;
        // cout << flow_fw << endl;
        //cout << !not_track << endl;
        cout << xt::filter(cur_y, !not_track) << endl;
        //// ~~~~xt::xarray<int> flow_bw1 = xt::index_view(flow_bw, xt::vstack(xtuple(xt::filter(cur_y, !not_track), xt::filter(cur_x, !not_track))));
        //flow_bw[!not_track] = flow_bw[cur_y[!not_track], cur_x[!not_track]];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (!not_track(i, j)) {
                    flow_bw(i, j, 0) = flow_bw(cur_y(i, j), cur_x(i, j), 0);
                    flow_bw(i, j, 1) = flow_bw(cur_y(i, j), cur_x(i, j), 1);
                }
            }
        }
        
        not_track = not_track | ((xt::square(xt::view(flow_fw, xt::all(), xt::all(), 0) + xt::view(flow_bw, xt::all(), xt::all(), 0)) +
            xt::square(xt::view(flow_fw, xt::all(), xt::all(), 1) + xt::view(flow_bw, xt::all(), xt::all(), 1))) >= check_thres);
        //track_cfd[cur_y[~not_track], cur_x[~not_track]] = prev_cfd[~not_track];
        //is_track[cur_y[~not_track], cur_x[~not_track]] = 1;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (!not_track(i, j)) {
                    track_cfd(cur_y(i, j), cur_x(i, j), 0) = prev_cfd(i, j, 0);
                    track_cfd(cur_y(i, j), cur_x(i, j), 1) = prev_cfd(i, j, 1);
                    is_track(cur_y(i, j), cur_x(i, j)) = 1;
                }
            }
        }

        xt::xarray <bool> not_flow = xt::zeros<bool>({ row, col});//;
        xt::xarray <float> dl_weights = xt::ones<float>({ row, col })*0.3;//;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                /*dl_weights(i, j) = 0.3;*/
                if (flow_fw(i,j,0) == 0  && flow_fw(i, j, 1) == 0 && flow_bw(i, j, 0) == 0  && flow_bw(i, j, 1) == 0) {
                    //not_flow(i, j) = 1;
                    dl_weights(cur_y(i, j), cur_x(i, j)) = 0.05;
                }
            }
        }
        xt::xarray <float> fusion_cfd = cur_cfd;
        //xt::filter(fusion_cfd, is_track) = xt::filter(dl_weights, is_track);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (is_track(i,j)) {
                    fusion_cfd(i, j) = dl_weights(i, j) * cur_cfd(i, j) + (1 - dl_weights(i, j)) * track_cfd(i, j);

                }
            }
        }

        xt::xarray <bool> index_certain = xt::zeros<bool>({ row, col });//;
        xt::xarray <bool> index_less01 = xt::zeros<bool>({ row, col });//;
        xt::xarray <bool> index_larger09 = xt::zeros<bool>({ row, col });//;

        index_certain = ((cur_cfd > 0.9) | (cur_cfd < 0.1)) * is_track;
        index_less01 = (dl_weights < 0.1) * index_certain;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (index_less01(i, j)) {
                    fusion_cfd(i, j) = 0.3 * cur_cfd(i, j) + 0.7 * track_cfd(i, j);
                }
            }
        }
        index_larger09 = (dl_weights >= 0.1) * index_certain;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (index_larger09(i, j)) {
                    fusion_cfd(i, j) = 0.4 * cur_cfd(i, j) + 0.6 * track_cfd(i, j);
                }
            }
        }

        prev_cfd = fusion_cfd;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                prev_gray.ptr<uchar>(i)[j] = cur_gray.ptr<uchar>(i)[j];
            }
        }
        Mat flowmap(row, col, CV_32FC1);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                //cout << fusion_cfd(i, j) << " ";
           //     if (fusion_cfd(i, j) > 50) {
           //         //cout << "yues" << endl;
           //         flowmap.ptr<float>(i)[j] = 1.0;
           //     }
           //     
          
           //     else  flowmap.ptr<float>(i)[j] = 0.0;
           //     //cout << (flowmap.ptr<float>(i)[j] == fusion_cfd(i, j)) << endl;
                flowmap.ptr<float>(i)[j] = fusion_cfd(i, j);
           }
           //cout << endl;
        }
        resize(flowmap, flowmap, Size(frame.cols, frame.rows), INTER_AREA);
        resize(scoremap, scoremap, Size(frame.cols, frame.rows), INTER_AREA);
        //for (int i = 0; i < h; i++) {
        //    for (int j = 0; j < w; j++) {
        //        //cout << fusion_cfd(i, j) << " ";
        //        if (flowmap.ptr<float>(i)[j] == 1.0) {
        //            cout << "yues" << endl;

        //        }
        //        else cout << "fuclk" << endl;
        //        //cout << (flowmap.ptr<float>(i)[j] == fusion_cfd(i, j)) << endl;
        //        //flowmap.ptr<float>(i)[j] = (fusion_cfd(i, j) > 0.01) ? 1 : 0;
        //    }
        //    cout << endl;
        //}
        Mat saveimg(frame.rows, frame.cols, CV_8UC3);
        Mat saveimg_f(frame.rows, frame.cols, CV_32FC3);
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < frame.rows; i++) {
                for (int j = 0; j < frame.cols; j++) {
                    saveimg.ptr<uchar>(i)[j * 3 + c] = scoremap.ptr<uchar>(i)[j] / 255.0 * frame.ptr<uchar>(i)[j * 3 + c] + (1 - scoremap.ptr<uchar>(i)[j]/255.0) * 255;
                    //saveimg_f.ptr<float>(i)[j * 3 + c] = flowmap.ptr<float>(i)[j] / 70.0 * frame.ptr<uchar>(i)[j * 3 + c] + (1 - flowmap.ptr<float>(i)[j] / 70.0 ) * 255;
                    saveimg_f.ptr<float>(i)[j * 3 + c] = flowmap.ptr<float>(i)[j] /60 * frame.ptr<uchar>(i)[j * 3 + c] + (1 - flowmap.ptr<float>(i)[j] / 60) * 255;
                }
            }
        }
        saveimg_f.convertTo(saveimg_f, CV_8UC3);

        cv::namedWindow("orig", 0);
        cv::resizeWindow("orig", 640, 480);
        cv::imshow("orig", frame);

 /*       cv::namedWindow("mask", 1);
        cv::resizeWindow("mask", 640, 480);*/
        cv::imshow("mask", saveimg);

 
        cv::namedWindow("after", 2);
        cv::resizeWindow("after", 640, 480);
        cv::imshow("after", saveimg_f);


        cv::waitKey(1);
    }
    cout << id / sum << endl;
    return 0;
}
