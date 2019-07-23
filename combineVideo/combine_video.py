import cv2
import os

from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int(x[:-11]))
    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def combine_video(pathIn, pathOut, fps):
    print("开始合成视频！")
    convert_frames_to_video(pathIn, pathOut, fps)
    print('视频已经合成完')


if __name__ == '__main__':
    pathIn = 'D:\\pythonpractice\\out_test\\'
    pathOut = 'D:\\pythonpractice\\video.avi'
    fps = 60.0