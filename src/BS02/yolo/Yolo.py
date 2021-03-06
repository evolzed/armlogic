import colorsys
import os
import sys
from datetime import datetime
from timeit import default_timer as timer
from ctypes import *

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image as PImage, ImageFont, ImageDraw

# from lib.GrabVideo import GrabVideo
# from lib.HikMvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX
from src.BS02.yolo.model import yolo_eval, yolo_body, tiny_yolo_body
from src.BS02.yolo.utils import letterbox_image

from keras.utils import multi_gpu_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

GPU_NUM = 1
gImgPath = None

# print(sys.path)


class YOLO(object):
    """神经网络YOLO类"""

    def __init__(self):
        # self.model_path = 'model_data/yolo_init.h5' # model path or trained weights path
        # self.anchors_path = 'model_data/yolo_anchors.txt'
        # self.classes_path = 'model_data/coco_classes.txt'
        # self.model_path = 'model_data/tiny_yolo_weights.h5' # model path or trained weights path

        # 自己的模型
        self.model_path = '../BS02/yolo/10000_trained_weights_final.h5' # model path or trained weights path
        self.anchors_path = '../BS02/yolo/tiny_yolo_anchors.txt'
        self.classes_path = '../BS02/yolo/22bottle_annotation_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (832, 832)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        # print(sys.path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        """
        载入模型，或构造模型和载入权重|Load model, or construct model and load weights

        :return:
                boxes:预测框,scores:得分,classes:类别
        """
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 载入模型，或构造模型和载入权重|Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            print(self.yolo_model.output)
            print(type(self.yolo_model.output))
            print(type(self.yolo_model.output[0]))
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 生成绘图边界框的颜色|Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # 为过滤的边界框生成输出张量目标|Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if GPU_NUM>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=GPU_NUM)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detectImage(self, image):
        """
        检测图片，画出识别框，输出识别出的类别

        :param image: 传入的图片
        :return: dataDict：构造出的图片数据集(image, timeCost, box:识别出的类别信息[(class, confidence, xmin, ymin, xmax, ymax),...])。
        """
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # TODO //
        if len(out_boxes) > 0:
            dataDict = {"isObj": True}
        #     try:
        #         image.save("")
        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='../BS02/yolo/font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        # 保存需要返回的数据的集合
        # dataDict = {"image": image}  # 更改位置到return之前
        dataDict = dict()
        boxList = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            # left, top, right, bottom == xmin, ymin, xmax, ymax
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            angle = None
            diameter = None

            centerX = None
            centerY = None
            deltaX = None
            deltaY = None
            speedX = None
            speedY = None
            trackID = None

            boxList.append([predicted_class, score, left, top, right, bottom,\
                            angle, diameter, centerX, centerY, trackID, deltaX, deltaY, speedX, speedY])

            dataDict["box"] = boxList
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        dataDict["timeCost"] = end - start
        # print(end - start)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dataDict["image"] = image
        return dataDict

    def closeSession(self):
        """
        关闭会话

        :return: None
        """
        self.sess.close()
