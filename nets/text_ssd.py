
''' SSD_Object Detection'''
''' 1
VGG16.py 主干特征提取网络
VGG16 imag3.shape=(300*300*3) 主干特征提取网络
Block1 300,300,3 -> 150,150,64 conv1_1(3*3,same,64)-->conv1_2(3*3,same,64)-->pool1(2*2,2*2)
Block2 150,150,64 -> 75,75,128 conv2_1(3*3,same,128)-->conv2_2(3*3,same,128)-->pool2(2*2,2*2)
Block3 75,75,128 -> 38,38,256  conv3_1(3*3,same,256)-->conv3_2(3*3,same,256)-->conv3_3(3*3,same,256)-->pool3(2*2,2*2)
Block4 38,38,256 -> 19,19,512  conv4_1(3*3,same,512)-->conv4_2(3*3,same,512)-->conv4_3(3*3,same,512)-->pool4(2*2,2*2)
Block5 19,19,512 -> 19,19,512  conv5_1(3*3,same,512)-->conv5_2(3*3,same,512)-->conv5_3(3*3,same,512)-->pool5(2*2,2*2)
FC6  19,19,512 -> 19,19,1024   Conv2D(3*3,dilation_rate=6*3,same,1024)
FC7  19,19,1024 -> 19,19,1024  Conv2D(1*1,same,1024)
Block6 19,19,512 -> 10,10,512  conv6_1(1*1,same,256)-->conv6_2(ZeroPadding2D)-->conv6_2(3*3,2*2,512)
Block7 10,10,512 -> 5,5,256    conv7_1(1*1,same,128)-->conv7_2(ZeroPadding2D)-->conv7_2(3*3,2*2,256)
Block8 5,5,256 -> 3,3,256      conv8_1(1*1,128)-->conv8_2(3*3,1*1,256)
Block9 3,3,256 -> 1,1,256      conv9_1(1*1,128)-->conv9_2(3*3,1*1,256)
'''

''' 2
ssd.py
SSD300(）流程
'conv4_3_norm'：
'conv4_3_norm_mbox_loc'，'conv4_3_norm_mbox_loc_flat'
'conv4_3_norm_mbox_conf'，'conv4_3_norm_mbox_conf_flat'
'conv4_3_norm_mbox_priorbox'

'fc7_mbox_loc'，'fc7_mbox_loc_flat'
'fc7_mbox_conf'，'fc7_mbox_conf_flat'
'fc7_mbox_priorbox'

'conv6_2_mbox_loc'，'conv6_2_mbox_loc_flat'
'conv6_2_mbox_conf'，'conv6_2_mbox_conf_flat'
'conv6_2_mbox_priorbox'

'conv7_2_mbox_loc'，'conv7_2_mbox_loc_flat'
'conv7_2_mbox_conf'，'conv7_2_mbox_conf_flat'
'conv7_2_mbox_priorbox'

'conv8_2_mbox_loc'，'conv8_2_mbox_loc_flat'
'conv8_2_mbox_conf'，'conv8_2_mbox_conf_flat'
'conv8_2_mbox_priorbox'

'conv9_2_mbox_loc'，'conv9_2_mbox_loc_flat'
'conv9_2_mbox_conf'，'conv9_2_mbox_conf_flat'
'conv9_2_mbox_priorbox'
'''


''' 3 ssd_layers.py
PriorBox详解
prior框高(1)-->网格中心(2)-->(1)+(2)得到框左上角和右下脚坐标-->映射到特征层--> 
框+Variance-->
'''

''' 4 如何解码预测结果并画出图像
ssd = SSD()
    r_image = ssd.detect_image(image)
        1 图片处理
            crop_img,x_offset,y_offset = letterbox_image()  # 防止图片失真
            photo = preprocess_input()
        2 预测 + 解码
            2.1 preds = self.ssd_model.predict(photo)  # [4,20,4]
            2.2 results = self.bbox_util.detection_out(preds, confidence_threshold=self.confidence)  # 解码
                2.2.1 mbox_loc,variances,mbox_priorbox,mbox_conf
                2.2.2 decode_bbox = self.decode_boxes() 解码
                      （1）获得先验框的宽与高、中心点
                      （2）根据公式求解预测框的宽与高、中心点
                      （3）转化为左上角和右下角四个点坐标
                2.2.3 取出得分高于confidence_threshold的框; 非极大抑制,取出在非极大抑制中效果较好的内容;
                      按照置信度进行排序,选出置信度最大的keep_top_k个
        3 筛选出其中得分高于confidence的框
        4 去灰条 
            boxes = ssd_correct_boxes()
        5 画框
'''



''' 5 labeling制作目标检测数据
pip3 install labelimg; labelimg;save translate;
'''
----------------------------------learning rate-----------------------------------------

''' 
# 1 阶段性下降 lr_n = lr_(n-1)*factor;  patience指学习率不变是可以容忍的代数;verbose=1 是否在终端打印出学习率的调整信息；
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)  

# 2 指数级下降
learning_rate = learning_rate_base* decay_rate^(global_epoch)

# 3 余弦退火衰减

'''

