import cv2
import keras
import numpy as np
import colorsys
import os
from nets import ssd
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from utils.utils import BBoxUtility,letterbox_image,ssd_correct_boxes
from PIL import Image,ImageFont, ImageDraw

class SSD(object):
    _defaults = {
        "model_path": 'model_data/ssd_weights.h5',     # 模型地址
        "classes_path": 'model_data/voc_classes.txt',  #  类别名称
        "model_image_size" : (300, 300, 3),            # 输入图片的大小
        "confidence": 0.5,                              # 置信度
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化ssd
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)     # 调用字典里的初始化对象
        self.class_names = self._get_class()     # 物体名称
        self.sess = K.get_session()
        self.generate()
        self.bbox_util = BBoxUtility(self.num_classes)
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    #---------------------------------------------------#
    #   获得所有的分、预测偏移量、先验框、每个类别对应的颜色
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算总的种类
        self.num_classes = len(self.class_names) + 1  # 类别+背景

        # 1 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.ssd_model = load_model(model_path,complie=False)
        except:
            self.ssd_model = ssd.SSD300(self.model_image_size, self.num_classes)
            self.ssd_model.load_weights(self.model_path, by_name=True)
        else:
            assert self.ssd_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.ssd_model.output)*(self.num_classes+5),\
                'Mismatch between model and given anchor and class size'


        self.ssd_model.summary()
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #  resize,加入灰条
        crop_img,x_offset,y_offset = letterbox_image(image, (self.model_image_size[0],self.model_image_size[1]))
        photo = np.array(crop_img,dtype = np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.reshape(photo,[1,self.model_image_size[0],self.model_image_size[1],3]))
        # 预测
        preds = self.ssd_model.predict(photo)

        # 解码、confidence、NMS、top_k
        results = self.bbox_util.detection_out(preds, confidence_threshold=self.confidence)
        # num1 = len(results[0])
        if len(results[0])<=0:
            return image

        # 筛选出其中得分高于confidence的框
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        # num2=tf.reduce_sum(top_label_indices)
        # print(num)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        # 去掉灰条
        boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]
        # 画框
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)-1]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)-1])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)-1])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def close_session(self):
        self.sess.close()
