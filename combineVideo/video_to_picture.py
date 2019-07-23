import cv2


def video_picture(video_file, from_dir):
    print("视频分解图片开始！")
    vc = cv2.VideoCapture(video_file)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        cv2.imwrite(from_dir + str(c) + '.jpg', frame)
        c = c + 1
        cv2.waitKey(1)
    vc.release()
    print("视频分解图片完成！")
