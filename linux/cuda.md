- 查看cuda 
  - cat /usr/local/cuda/version.txt
- 查看cudnn 
  - cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
- 查看显卡驱动版本
  - cat /proc/driver/nvidia/version 或输入nvidia-smi
  
- 版本切换
  - 安装多版本cuda后在/usr/local/目录下查看自己安装的cuda版本
  - 使用一个symbolic link 指向想使用的GPU：ln -s /usr/local/cuda-9.1 /usr/local/cuda
  - cuda的环境变量设置成这个软链接的，实现解耦 具体参考 https://blog.csdn.net/yinxingtianxia/article/details/80462892


- 监控GPU
  - nvidia-smi
  - watch -n 1 nvidia-smi 每隔1s刷新一次
  
  

- 下载cuda
  - 官网https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal
  - 复制链接 使用迅雷下载速度较快

- 指定gpu训练
  - os.environ["CUDA_VISIBLE_DEVICES"] = "2"

