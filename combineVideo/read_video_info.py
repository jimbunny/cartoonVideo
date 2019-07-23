import cv2


def read_video_info(file):
    cap = cv2.VideoCapture(file)

    # 获取视频分辨率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 输出文件编码，Linux下可选X264
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # 视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("视频size:" + str(size))
    print("视频编码:" + str(fourcc))
    print("视频的FPS:" + str(fps))
    return size, fourcc, fps
