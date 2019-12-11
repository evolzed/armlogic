# 基于keras的yolov3说明

## 环境要求：

1. 系统win10 、ubuntu18.04、ubuntu16.04均可

2. python==3.6.5

3. tensorflow-gpu==1.6.0或tensorflow-gpu==1.9.0（gpu版本需要配置cuda）
  + Tips：安装cuda：查看自己的gpu是否支持cuda[点击查看我的GPU是否支持安装cuda](https://developer.nvidia.com/cuda-gpus)，如果不支持，只能使用cpu版本的tensorflow，同样安装tensorflow==1.6.0或tensorflow==1.9.0

4. python-opencv==3.4.2.16（尽量不要使用最新4+版本）

5. numpy==1.16.5（win下需要1.16.5+mkl）

6. keras==2.1.5（务必使用该版本，否则报错）

## 使用：

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```python
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo.py   OR   python yolo_video.py [video_path] [output_path(optional)]
```

​	由于GPU限制（全网络的yolo运行显存至少12G+），对于4G的GTX980我们用Tiny-YOLOv3，按照上面的方法下载tiny-weights文件和tiny-cfg文件，注意修改yolo.py中的model和anchor路径。

4. MultiGPU usage is an optinal. Change the number of gpu and add gpu device id

## 训练模型

1. 构造自己的annotation文件和class文件

   【注意生成的txt的格式】：

   One row for one image;  
   Row format: `image_file_path box1 box2 ... boxN`;  
   Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
   For VOC dataset, try `python voc_annotation.py`  
   Here is an example:

   ```python
   path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
   path/to/img2.jpg 120,300,250,600,2
   ...
   ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
   The file model_data/yolo_weights.h5 is used to load pretrained weights.
3. Modify train.py and start training.  
   `python train.py`  
   Use your trained weights or checkpoint weights in yolo.py.  
   Remember to modify class path or anchor path.

##  cfg配置参数含义

```python
[net]
# Testing
# batch=1
# subdivisions=1
# Training

batch=32 # 一批训练样本的样本数量，每batch个样本更新一次参数
subdivisions=16  # batch/subdivisions作为一次性送入训练器的样本数量如果内存不够大，将batch分割为subdivisions个子batch。

上面这两个参数如果电脑内存小，则把batch改小一点，batch越大，训练效果越好subdivisions越大，可以减轻显卡压力

width=416
height=416
channels=3
# 以上三个参数为输入图像的参数信息 width和height影响网络对输入图像的分辨率，从而影响precision，只可以设置成32的倍数

momentum=0.9  # DeepLearning1中最优化方法中的动量参数，这个值影响着梯度下降到最优值得速度 
decay=0.0005  # 权重衰减正则项，防止过拟合
angle=0   # 通过旋转角度来生成更多训练样本
saturation = 1.5  # 通过调整饱和度来生成更多训练样本
exposure = 1.5  # 通过调整曝光量来生成更多训练样本
hue=.1  # 通过调整色调来生成更多训练样本

learning_rate=0.001 # 学习率决定着权值更新的速度，设置得太大会使结果超过最优值，太小会使下降速度过慢。如果仅靠人为干预调整参数，需要不断修改学习率。刚开始训练时可以将学习率设置的高一点，而一定轮数之后，将其减小在训练过程中，一般根据训练轮数设置动态变化的学习率。刚开始训练时：学习率以 0.01 ~ 0.001 为宜。一定轮数过后：逐渐减缓。接近训练结束：学习速率的衰减应该在100倍以上。学习率的调整参考https://blog.csdn.net/qq_33485434/article/details/80452941

burn_in=1000  # 在迭代次数小于burn_in时，其学习率的更新有一种方式，大于burn_in时，才采用policy的更新方式

max_batches = 500200  # 训练达到max_batches后停止学习
policy=steps  # 这个是学习率调整的策略，有policy：constant, steps, exp, poly, step, sig, RANDOM，constant等方式

steps=4000,4500   # 下面这两个参数steps和scale是设置学习率的变化，比如迭代到4000次时，学习率衰减十倍。4500次迭代时，学习率又会在前一个学习率的基础上衰减十倍
scales=.1,.1

[convolutional]
batch_normalize=1  # 是否做BN
filters=16   # 输出特征图的数量
size=3   # 卷积核的尺寸
stride=1  # 做卷积运算的步长
pad=1  # 如果pad为0,padding由 padding参数指定。如果pad为1，padding大小为size/2，padding应该是对输入图像左边缘拓展的像素数量
activation=leaky  # 激活函数的类型

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=1

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

###########

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=27  # 每一个[region/yolo]层前的最后一个卷积层中的filters=num(yolo层个数)*(classes+5)5的意义是5个坐标，论文中的tx,ty,tw,th,to

activation=linear



[yolo]   #  在yoloV2中yolo层叫region层
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319  #   anchors是可以事先通过cmd指令计算出来的，是和图片数量，width,height以及cluster(应该就是下面的num的值，即想要使用的anchors的数量)相关的预选框，可以手工挑选，也可以通过k means 从训练样本中学出

classes=4
num=6    #  每个grid cell预测几个box,和anchors的数量一致。当想要使用更多anchors时需要调大num，且如果调大num后训练时Obj趋近0的话可以尝试调大object_scale
jitter=.3  #  利用数据抖动产生更多数据，YOLOv2中使用的是crop，filp，以及net层的angle，flip是随机的，jitter就是crop的参数，tiny-yolo-voc.cfg中jitter=.3，就是在0~0.3中进行crop
ignore_thresh = .7  # 决定是否需要计算IOU误差的参数，大于thresh，IOU误差不会夹在cost function中
truth_thresh = 1
random=1  # random设置成1，可以增加检测精度precision；如果为1，每次迭代图片大小随机从320到608，步长为32，如果为0，每次训练大小与输入大小一致

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 8

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=27
activation=linear

[yolo]
mask = 1,2,3
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=4
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
```

## 准备数据

1. 准备数据，图像的(.jpg)数据的目录在OpenLabeling/main/input/下，标注的文件(.txt)，存放在OpenLabeling/main/output/YOLO_darknet/下。注意标注信息的格式是否正确。注意，每个txt文件对应一个jpg文件。文件要同名。注意需要对txt进行处理，生成annotation.txt文件。

2. 调用合并文件的脚本，使用cd 到main目录进行python merge_annotation.py，将会在output目录下生成my_annotation.txt：

   `x_min,y_min,x_max,y_max,class_id` (no space).

   如果是多个目标物:放在一排，不同的类之间用空格隔开，坐标之间用逗号隔开。

3. 训练模型中指定my_annotation.txt即可

## 开始训练

+ ##根据所需要的类别数调整好基本参数后，编译预训练的weights文件，注意修改文件路径：

1. annotation_path = "yourAnnotationFilePath"
2. log_dir = "yourLogsFilesPath"
3. classes_path = "yourClassesFilePath"
4. anchors_path = "anchorsFilePathYouUsed"
5. input_shapt = (416, 416) # 默认416， 如需提高小物体检测效果，可以改成其他32倍数的数字，如832，或608
6. val_split = 0.1  # 该参数调整数据集中的验证机比例，0.1代表验证集占整个数据集的10%， 小数据集可以为0.2或0.3，大数据集设置为0.1或0.05，根据数据集大小适当调整
7. 对模型进行编译，在compile()方法优化器参数optimizer中设置采用Adam算法，学习率lr不要设置过大，从1e-3(0.001)开始，损失函数采用yolo_loss。
8. 第一次迭代中的batch_size = 32,如果运行时报错run out of memory的话，说明一次进入到模型的数据太多，显存吃不下。此时可以适当调整batch_size为16或者8，如果显存够大，可以适当调整该值为64甚至128
   然后调整fit_generator()中的epochs数值，initial_epoch=0,此为第一次迭代，从0开始
9. 第二阶段训练，同样，需要调整合适的batch_size，为了获得更优梯度下降效果，需要设置一个比上一次低的学习率。当然了，该阶段训练要设置学习率自动下降。
10. 调整epochs为训练总共的迭代次数，需要注意的是，initial_epoch参数需要设置成第一阶段的epochs值
11. 第二阶段的训练在callbacks参数中加入了reduce_lr，设置学习率下降梯度。另外一个early_stopping，当损失值在限定次数内不再出现最小时，将触发early_stopping，训练终止
12. 最终，训练好的.h5文件将保存在log_dir中。用于检测物体时给模型使用

## 模型效果实时检测

1. 运行keras-yolo3-02下面的yolo.py文件python yolo.py。
2. 其中的`detect_img()`方法，封装了对图片进行识别。`detect_video()`中封装了调用摄像机获取实时数据和加载视频进行识别， 通过`video_path`参数进行切换（0：实时检测，如果是视频路径，则对视频进行载入并识别）。
3. 需要对控制输出参数的话，修改`detect_image()`中的参数逻辑，根据需求对输出参数的格式和数量类型进行重新改写封装。
