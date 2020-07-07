from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.preprocessing import image
from nets.ssd import SSD300
from nets.ssd_training import MultiboxLoss,Generator
from utils.utils import BBoxUtility
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import cv2
import keras
import os
import sys
if __name__ == "__main__":
    log_dir = "logs/"
    annotation_path = '2007_train.txt'
    
    NUM_CLASSES = 21
    input_shape = (300, 300, 3)
    # 读取先验框
    priors = pickle.load(open('model_data/prior_boxes_ssd300.pkl', 'rb'))
    # 处理先验框的类函数
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('model_data/ssd_weights.h5', by_name=True, skip_mismatch=True)

    # 训练参数设置
    logging = TensorBoard(log_dir=log_dir)
    # 保存训练过程的参数
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)  # 学习率阶段性下降
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    BATCH_SIZE = 1
    # 数据生成器，在训练过程中不断传入数据给模型
    gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES)

    for i in range(21):
        model.layers[i].trainable = False
    # 粗略训练
    if True:
        # MultiboxLoss() 计算损失
        model.compile(optimizer=Adam(lr=1e-4),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        # gen.generate(True)生成标签
        model.fit_generator(gen.generate(True),
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=15, 
                initial_epoch=0,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])  # 回调函数
    # 精细训练
    if True:
        model.compile(optimizer=Adam(lr=1e-5),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=30, 
                initial_epoch=15,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(21):
        model.layers[i].trainable = True
    if True:
        model.compile(optimizer=Adam(lr=1e-6),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=50, 
                initial_epoch=30,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])