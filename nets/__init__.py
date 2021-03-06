'''
训练步骤

1、本文使用VOC格式进行训练。
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。
4、在训练前利用voc2yolo3.py文件生成对应的txt。
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
6、就会生成对应的2007_train.txt，每一行对应其图片位置及其真实框的位置。
7、在训练前需要修改model_data里面的voc_classes.txt文件，需要将classes改成你自己的classes。
8、运行train.py即可开始训练。

tensorflow-gpu==1.13.1
keras==2.1.5
'''
