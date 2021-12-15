# LDF


## 去背景应用


- LDFbased-merger.py 使用opencv 切帧和合帧

- LDFbased_merger_ffmpeg.py 使用 FFMPEG.

- process_all_video.sh 对input_video目录下每个目标进行指定背景替换

- Result: 简单场景下，用显著性目标检测代替了alpha matting做的背景替换
