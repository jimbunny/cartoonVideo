#!--*-- coding:utf-8 --*--
# import numpy as np
import tensorflow as tf
import cv2

# Deeplab Demo

import os
import tarfile

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile
from six.moves import urllib

import tensorflow as tf


class DeepLabModel(object):
    """
 加载 DeepLab 模型；
 推断 Inference.
 """
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """
  Creates and loads pretrained deeplab model.
  """
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """
  Runs inference on a single image.

  Args:
  image: A PIL.Image object, raw input image.

  Returns:
  resized_image: RGB image resized from original input image.
  seg_map: Segmentation map of `resized_image`.
  """
        width, height = image.size
        # resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        resize_ratio = 1
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def run2(self, image):
        width = image.shape[1]
        height = image.shape[0]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        b, g, r = cv2.split(image)
        img_rgb = cv2.merge([r, g, b])
        resized_image = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_CUBIC)
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3
    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]


def load_model():
    model_path = './deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'#'deeplab_model.tar.gz'
    MODEL = DeepLabModel(model_path)
    print('model loaded successfully!')
    return MODEL


model = load_model()


def combine_backend(pic, bk, to_dir, i):
    src = cv2.imread(pic)

    # Read image with Image
    src_view = cv2.imread(bk)

    resized_im, seg_map = model.run2(src)
    resized_view = cv2.resize(src_view,(resized_im.shape[1],resized_im.shape[0]))
    resized_view = cv2.medianBlur(resized_view,11)
    # seg_image = label_to_color_image(seg_map).astype(np.uint8)
    # seg_map = cv2.GaussianBlur(np.uint8(seg_map),(11,11),0)
    src_resized = cv2.resize(src,(resized_im.shape[1],resized_im.shape[0]))
    # seg_image = cv2.GaussianBlur(seg_image,(11,11),0)
    bg_img = np.zeros_like(src_resized)

    # 复制背景
    bg_img[seg_map == 0] = src_resized[seg_map == 0]

    blured_bg = cv2.GaussianBlur(bg_img,(11,11),0)
    result = np.zeros_like(bg_img)

    # 合成
    result[seg_map > 0] = resized_im[seg_map > 0]
    result[seg_map == 0] = blured_bg[seg_map == 0]

    # 背景变换与合成
    result_2 = np.zeros_like(bg_img)
    result_2[seg_map > 0] = src_resized[seg_map > 0]
    result_2[seg_map == 0] = resized_view[seg_map == 0]

    # cv2.imwrite('D:\\pythonpractice\\bkkkkkkkkkkkkkk.jpg', result_2)
    cv2.imwrite(to_dir + str(i) + '.jpg', result_2)
    # cv2.imshow('src',src)
    # cv2.imshow('resized_im',resized_im)
    # cv2.imshow("seg_image",seg_image)
    # cv2.imshow('bg_image',bg_img)
    # cv2.imshow('blured_bg',blured_bg)
    # cv2.imshow('result',result)
    # cv2.imshow('result_2', result_2)
    #
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def combine_bk(bk, img_dir, to_dir):
    print("开始照片换背景！")
    all_images = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
    a = 0
    for i in all_images:
        print("正在换背景第" + str(a) + '张')
        a += 1
        combine_backend(i, bk, to_dir, a)
    print("照片换背景完成！")


if __name__ == '__main__':
    file1 = 'D:\\pythonpractice\\from\\1.jpg'
    file2 = 'D:\\pythonpractice\\from\\1.jpg'
