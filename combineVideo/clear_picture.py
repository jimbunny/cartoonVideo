# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os


def clear(file, i, to_dir):
    # 加载图像
    image = cv2.imread(file)
    # 自定义卷积核
    kernel_sharpen_1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])
    kernel_sharpen_2 = np.array([
    [1, 1, 1],
    [1, -7, 1],
    [1, 1, 1]])
    kernel_sharpen_3 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [-1, -1, -1, -1, -1]]) / 8.0
    # 卷积
    # output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
    # output_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
    output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
    cv2.imwrite(to_dir + str(i) + '.jpg', output_3)
    # 显示锐化效果
    # cv2.imshow('Original Image', image)
    # cv2.imshow('sharpen_1 Image', output_1)
    # cv2.imshow('sharpen_2 Image', output_2)
    # cv2.imshow('sharpen_3 Image', output_3)
    # 停顿
    # if cv2.waitKey(0) & 0xFF == 27:
    #     cv2.destroyAllWindows()


def clear2(file, i, to_dir):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(file, -1, kernel=kernel)
    cv2.imwrite(to_dir + str(i) + '.jpg', dst)


def clear_picture(img_dir, to_dir):
    print("开始锐化照片！")
    all_images = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
    all_images.sort(key=lambda x: int(x.replace(img_dir, "")[:-4]))
    a = 0
    for i in all_images:
        print("正在锐化第" + str(a) + '张')
        a += 1
        clear(i, a, to_dir)
    print("照片锐化完成！")
