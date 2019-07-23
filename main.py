from combineVideo.download_video import parser_video
from combineVideo.video_to_picture import video_picture
from combineVideo.read_video_info import read_video_info
from combineVideo.clear_picture import clear_picture
from combineVideo.combine_bk import combine_bk
from combineVideo.combine_video import combine_video

url = 'http://v.douyin.com/By1ten/'
download_url = 'http://videotool.com.cn/video/loadNative?videoUrl='  # 解析后下载地址
download_file = "D:\\pythonpractice\\download\\videoplayback.mp4"  # 下载的视频保存路径
from_dir = "D:\\pythonpractice\\from2\\"  # 视频分解图片保存路径
to_dir = "D:\\pythonpractice\\to2\\"  # 动画渲染后图片保存路径
clear_dir = "D:\\pythonpractice\\clear_to2\\"  # 锐化后图片保存路径
bk = "D:\\pythonpractice\\bk.jpg"  # 替换背景图片
pathIn = 'D:\\pythonpractice\\test2\\test_output\\'  # 合成动画的图片路径
pathOut = 'D:\\pythonpractice\\video_new.avi'  # 最终的成果
fps = 23.976023976023978

if __name__ == '__main__':
    url = 'http://v.douyin.com/B71E4e/'
    # 去除水印下载视频
    parser_video(url, download_url, download_file)

    # 将视频按帧数分解图片
    video_picture(download_file, from_dir)

    # 读取视频信息
    read_video_info(download_file)

    # 锐化图片
    clear_picture(from_dir, clear_dir)

    # 替换照片背景
    combine_bk(bk, clear_dir, to_dir)

    # 将图片合成动画
    # https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch
    # python test.py --input_dir /root/CartoonGAN-Test-Pytorch-Torch/test_img --style Hosoda --gpu -1

    # 将处理好的图片合成视频
    combine_video(pathIn, pathOut, fps)

    # 音频视频合成
    # ffmpeg -i test.mp3 -ss 00:00:00 -t 00:00:06 -acodec copy test2.mp3
    # ffmpeg - i video.avi - i test2.mp3 - c copy video2.avi
    # ffmpeg -i output.mp4 -c:a aac -b:a 160k output2.mp4
    # ffmpeg -i video.avi -i test2.mp3  -c:v copy -c:a aac -strict experimental output.mp4
    # ffmpeg -i 3.mp4 -vn -y -acodec copy 3.aac/m4a