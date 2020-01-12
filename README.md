# BottleSort0.2设计说明书

# Objective：Track Function & Optimize Vision Efficiency

# 1.**环境搭建**

## 1.1 硬件环境

* 运行平台： PC/Win10_x86_64**&**Linux
* 工业相机
  + 品牌：HIKVision 
  + 接口协议：Gigabit Ethernet（GigE协议）
  + 型号:MV-CE013-50GC
  
## 1.2 软件环境
* Ubuntu 18.04
* PyCharm 2018.3.5
* python==3.6.5
* opencv==4.4.1
* tensorflow-gpu==1.6.0或tensorflow-gpu==1.9.0（gpu版本需要配置cuda）
  + Tips：安装cuda：查看自己的gpu是否支持cuda[点击查看我的GPU是否支持安装cuda](https://developer.nvidia.com/cuda-gpus)，如果不支持，只能使用cpu版本的tensorflow，同样安装tensorflow==1.6.0或tensorflow==1.9.0
* python-opencv==3.4.2.16（尽量不要使用最新4+版本）
* numpy==1.16.5（必须要用1.16版本，1.17报错；win下需要1.16.5+mkl）
* keras==2.1.5（务必使用该版本，否则报错）
* PIL第三方库
* 标准库：os模块、ctype(调用c/c++代码)、datetime(日期模块)、colorsys(转换模型模块)、timeit(测试程序运行时间)
* [GitHub](https://github.com/evolzed/armlogic)
## 1.3 通讯接口
* Ethernet
* RS485

----
# 2.**定义及规则**

## 2.1 文件夹
* project_root/
  * lib/          #库
  * docs/         #技术文档
  * src/          #源代码
  * test/         #测试
  * README.md     
<<<<<<< HEAD
  * LICENSE.md     
  
## 2.2 系统流程图
![FlowChart](http://192.168.0.203:8088/armlogic/BottleSort/raw/Track/docs/pic/FlowChart/Track_20200102.png)

## 2.3 数据变量命名规则

|   类型   | 命名举例 |              描述               | 第一元素 | 第二元素 |  第三元素 |  第四元素 |  第五元素 |
| :------: | :------: | :-----------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| global |  gState  | #1 = 初始阶段, 2 = 运行阶段, 3 = 停止阶段 | int gState |||||
| library |  airDict  | #User input air pressure nozzle locations | int x | int y | int z | int type ||
| library |  bgDict  | #address to image file location and processed flag | #%.JPG | bool processed ||||
| global |  gDir  | #direction of the moving belt | int gDir | ||||
| library |  bottleDict  | #sorted bottle information | {"image":imagedata,"box":[(bottletype1, confidence, xmin, ymin, xmax, ymax),(bottletype2, confidence, xmin, ymin, xmax, ymax),···]，"bgTimeCost":time,"timeCost":time,"nFrame":camNumFrame} |  |  | | |
| library | trackDict | #target track information | {"target":[(UUID, trackFlag, postion, speed, angle, type, typeCounter, nFrame, bgTimeCost, timeCost, trackTimeCost)]} | | | | |
| path | bgPic | #file location for bgPic ||||||

## 2.4 功能包文档填写说明

|   Class   | Function |           Description          | Input | Output | Return |
| :------: | :------: | :-----------------------------: | :----: | :----: | :----: |
| Main |  | 主方法，系统流程 | gState |  |  |
<<<<<<< HEAD
| vision |  generate  | 返回预测框列表，评分列表，类别列表, 使用load_model()、yolo_eval() | | | |
| vision |  connectCam  | 获取检测到的设备编号，连接设备GrabVideo.getDeviceNum()、GrabVideo.connectCam() | | | |
| vision |  grabVideo  | 获取相机的视频流,利用封装好的GrabVideo包进行获取 | | %bgPic.JPG | |
| Camera |  getImage  | get a frame of opencv format from camra | %_cam  %_data_buf %_nPayloadSize|  | %frame|
| Camera | getCamFps  | get  fps of camera output in str type  | %nFrameNum  |  | %fps|
| vision |  loadYolo  | Tiny-Yolov3模型参数初始化(包含model_path、anchors_path、classes_path等), 调用generate()方法，初始化boxes，scores， classes | %.py %.pt | | |
| vision |  detectImage  | 检测输入图像的函数, 调用letterbox_image():不损坏原图尺寸比例进行填充；PIL下的ImageDraw模块中的Draw()->对图像进行画框标注, 将数据流传给yoloCNN，cv2.cvtColor()[色彩空间转换]、PIL.Image()[转换成网络需要的imageObject]; | | | |
| vision |  studyBackgroundFromCam  | get 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk| %cam | |  |
| vision |  avgBackground  | learn the backgroud from disk then  accumulate every frame difference,accumulate every frame  | %img |  |  |
| vision |  createModelsfromStats  | average the frame and frame difference to get the background model| %I %dst | bottleDict||
| vision |  backgroundDiff  | use the background to segment the frame pic| %src %dst | ||
| Vision | checkState | [1:init 2：run 3：stop], 停止网络，关闭相机驱动 |  | ||
| Vision | getBeltSpeed() | \#get belt speed direction and valu e,pixel per second | bottleDict | |beltSpeed|
| Vision | getBottlePos() | \#get belt speed direction and valu e,pixel per second | bottleDict | |bottleDetail|
| Vision | getBottleID() | \#get bottle ID by track and beltSpeed | bottleDict | beltSpeed |bottleID|
| Track | createTarget() | 创建新的trackDict元素 |  | trackDict ||
| Track | updateTarget() | 更新trackDict内的元素 |  | trackDict ||
=======
| Vision |  generate  | 返回预测框列表，评分列表，类别列表, 使用load_model()、yolo_eval() | | | |
| Vision |  connectCam  | 获取检测到的设备编号，连接设备GrabVideo.getDeviceNum()、GrabVideo.connectCam() | | | |
| Vision |  grabVideo  | 获取相机的视频流,利用封装好的GrabVideo包进行获取 | | %bgPic.JPG | |
| Camera |  getImage  | get a frame of opencv format from camra | %_cam  %_data_buf %_nPayloadSize|  | %frame|
| Camera | getCamFps  | get  fps of camera output in str type  | %nFrameNum  |  | %fps|
| Vision |  loadYolo  | Tiny-Yolov3模型参数初始化(包含model_path、anchors_path、classes_path等), 调用generate()方法，初始化boxes，scores， classes | %.py %.pt | | |
| Vision |  detectImage  | 检测输入图像的函数, 调用letterbox_image():不损坏原图尺寸比例进行填充；PIL下的ImageDraw模块中的Draw()->对图像进行画框标注, 将数据流传给yoloCNN，cv2.cvtColor()[色彩空间转换]、PIL.Vision()[转换成网络需要的imageObject]; | | | |
| ImgProc |  studyBackgroundFromCam  | get 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk| %cam | |  |
| ImgProc |  avgBackground  | learn the backgroud from disk then  accumulate every frame difference,accumulate every frame  | %img |  |  |
| ImgProc |  createModelsfromStats  | average the frame and frame difference to get the background model| %I %dst | bottleDict||
| ImgProc |  backgroundDiff  | use the background to segment the frame pic| %src %dst | ||
| Vision |  checkState  | [1:init 2：run 3：stop], 停止网络，关闭相机驱动
|  Vision   | getBeltSpeed()  | #get belt speed direction and valu e,pixel per second   |   bottleDict                     |  | beltSpeed |
|  ImgProc   | getBottlePose()  | #get belt speed direction and valu e,pixel per second   |bottleDict |  | bottleDetail |
|  ImgProc   | ()  |   | |  |  |
|  Vision   | getBottleID()  | #get bottle ID by track and beltSpeed   |   bottleDict                     | beltSpeed | bottleID |



| Class |    Function    |      Description      | Input |  Output   | Return |
| :---: | :------------: | :-------------------: | :---: | :-------: | :----: |
| Track | createTarget() | 创建新的trackDict元素 |       | trackDict |        |
| Track | updateTarget() | 更新trackDict内的元素 |       | trackDict |        |

>>>>>>> e37de98efe0758be20affa29c5d30d0aad575bb4


----

## 3.测试BS0.1

#### 测试环境

|   名称   |       详情        |
| :------: | :---------------: |
|   硬件   |        TX2        |
|   系统   |       Liunx       |
|   语言   |     python3.6     |
| 所属模块 | detectSingleImage |
|   时间   |    2019-12-23     |



#### 功能测试

|           用例标题           |                           预期                            |                             实际                             |                           修改建议                           | 优先级 |
| :--------------------------: | :-------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----: |
|         Camera的启动         | Camera正常启动成功；如果无法启动。控制台输出错误提示信息  | 1.Camera启动过慢。2.没有其他用户使用Camera时会正常启动，如果在启动时其他用户正在使用Camera，则如果无法启动。控制台输出错误提示信息。3.输入启动命令会提示初始化失败，需要多次重试启动 | 1.开启Camera之前确保没有其他用户在使用Camera。2.Camera在命令输入后在短时间正常启动 |   中   |
|           背景学习           | 初始背景全部涂黑,当物体出现在视野范围，显示出物体的轮廓。 | 实际结果与预期相符，但当开启Camera进行背景学习时如果当前视野中有塑料品或其他物品，初始背景同样会全部涂黑 |                             暂无                             |   低   |
|  接受bgLearn返回过来的图片   |             接收bgLearnb背景学习后的物体图片              |             正常接收bgLearnb背景学习后的物体图片             |                             暂无                             |   低   |
| 利用神经网络识别塑料瓶的类别 |                   正确识别塑料瓶的类别                    | 1.视野中出现的塑料瓶，识别的召回率不高。2.识别的准确率高低不一。3.会把视野中其他物品识别成瓶子。4.瓶子数量密集时识别的标注框会变大造成多个瓶子一个标注框 | 提高识别塑料瓶的召回率，瓶子数量密集时标注框独立标注平面的每一个塑料瓶 |   高   |
|         返回坐标信息         |                返回识别到塑料瓶的坐标信息                 |                 正确返回识别到物体的坐标信息                 |                             暂无                             |   低   |
|         捕捉到的帧数         |                   捕捉到的帧数每次波动正常+1                   |                捕捉到的帧数（nFrame）没有波动                |            优化代码逻辑，使其捕捉到的帧数正常波动            |   中   |
|         Camera的关闭         |          当输入Camera的关闭按键，Camera正常关闭           | 1.在Camera开启的页面输入"q"命令,Camera经常无法正常关闭。2.过程中会有一定延迟后关闭或是一直按若干次"q"才可以关闭。3.如果一直s输入"q"还未关闭会出现界面卡死。。 |             保证输入规定的命令时,Camera正常关闭              |   中   |



#### 性能测试

| 用力标题 |                             预期                             |                             实际                             |                 修改建议                 |  优先级  |
| :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------------------------: | :--: |
|   FPS    | 摄像头的实际的FPS受千兆网传输速率影响较大。目前预期最高可能只达到20帧左右 |                  TX2运行实际达到的帧数只有1                  |             提高运行时的帧率             |  中  |
| 识别时间 |                  物体识别时间达到项目的需求                  | ![识别时间](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/%E8%AF%86%E5%88%AB%E6%97%B6%E9%97%B4.jpg)塑料瓶识别时间 |       减短识别塑料瓶所用的时间成本       |  高  |
|  准确率  |                物体识别的准确率达到项目的需求                | ![准确率](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/%E5%87%86%E7%A1%AE%E7%8E%87.jpg)物体识别的准确率低 | 优化代码逻辑，提高识别塑料品的整体准确率 |  高  |
|  稳定性  |             整体稳定性正常，不影响项目的正常需求             | 识别塑料瓶的稳定性低![瓶子数量](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/%E7%93%B6%E5%AD%90%E6%95%B0%E9%87%8F.jpg) |          提高塑料瓶识别的稳定性          |  高  |

---------------------------------------------------------------------------------------------------------------

### 理想状态下固定数量的瓶子识别情况

|      用例标题       |  测试结构图   |                           测试结果                           |              修改建议              |
| :-----------------: | :-----------: | :----------------------------------------------------------: | :--------------------------------: |
| 1个瓶子时的识别时间 |     ![识别时间](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/1time.jpg)      |                       识别时间用时过长                       |       减少识别物体花费的时间       |
| 1个瓶子时的识别数量 |     ![识别数量](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/1num.jpg)       |   固定数量时的瓶子识别数量不固定，会把其他物体也识别成瓶子   | 提高识别瓶子的精准度，只识别瓶子。 |
| 1个瓶子时识别准确率 |     ![识别准确率](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/1acc.jpg)     | 标准环境中放置固定数量的环境，每帧识别到瓶子的准确率会不稳定 |          提高识别的准确率          |
| 2个瓶子时的识别时间 |     ![识别时间](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/2time.jpg)      |                       识别时间用时过长                       |       减少识别物体花费的时间       |
| 2个瓶子时的识别数量 |     ![识别数量](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/2num.jpg)       |   固定数量时的瓶子识别数量不固定，会把其他物体也识别成瓶子   | 提高识别瓶子的精准度，只识别瓶子。 |
| 2个瓶子时识别准确率 |     ![识别准确率](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/2acc.jpg)     | 标准环境中放置固定数量的环境，每帧识别到瓶子的准确率会不稳定 |          提高识别的准确率          |
| 3个瓶子时的识别时间 |     ![识别时间](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/3time.jpg)      |                       识别时间用时过长                       |       减少识别物体花费的时间       |
| 3个瓶子时的识别数量 |     ![识别数量](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/3num.jpg)       |   固定数量时的瓶子识别数量不固定，会把其他物体也识别成瓶子   | 提高识别瓶子的精准度，只识别瓶子。 |
| 3个瓶子时识别准确率 |     ![识别准确率](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/3acc.jpg)     | 标准环境中放置固定数量的环境，每帧识别到瓶子的准确率会不稳定 |          提高识别的准确率          |
| 4个瓶子时的识别时间 |     ![识别时间](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/4time.jpg)      |                       识别时间用时过长                       |       减少识别物体花费的时间       |
| 4个瓶子时的识别数量 |     ![识别数量](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/4num.jpg)       |   固定数量时的瓶子识别数量不固定，会把其他物体也识别成瓶子   | 提高识别瓶子的精准度，只识别瓶子。 |
| 4个瓶子时识别准确率 |     ![识别准确率](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/4acc.jpg)     | 标准环境中放置固定数量的环境，每帧识别到瓶子的准确率会不稳定 |          提高识别的准确率          |
| 5个瓶子时的识别时间 |     ![识别时间](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/5time.jpg)      |                       识别时间用时过长                       |       减少识别物体花费的时间       |
| 5个瓶子时的识别数量 |     ![识别数量](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/5num.jpg)       |   固定数量时的瓶子识别数量不固定，会把其他物体也识别成瓶子   | 提高识别瓶子的精准度，只识别瓶子。 |
| 5个瓶子时识别准确率 |     ![识别准确率](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/5acc.jpg)     | 标准环境中放置固定数量的环境，每帧识别到瓶子的准确率会不稳定 |          提高识别的准确率          |

----
#### 测试环境
| 名称     | 详情                            |
| -------- | --------------------------------------- |
| 硬件     | TX2，PC |
| 系统    | PyCharm 2016.1.5 |
| 语言     | python3.6 |
| 测试模块 | MovePlanning     |
| 时间 | 2019/12/31 |

#### 测试流程

|                         采集图片数据                         |                         测量参数值                          |             测试结果             |
| :----------------------------------------------------------: | :---------------------------------------------------------: | :------------------------------: |
| ![第1张图片数据](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/collection1.bmp) | 第一个瓶子：626，433，80，60；第二个瓶子：813，438，-80，60 | 机械臂吸盘精准到达标定的中心位置 |
| ![第2张图片数据](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/collection2.bmp) | 第一个瓶子：656，463，-3，60；第二个瓶子：845，458，88，60  | 机械臂吸盘精准到达标定的中心位置 |
| ![第3张图片数据](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/collection3.bmp) | 第一个瓶子：639，448，-42，60；第二个瓶子：825，434，50，60 | 机械臂吸盘精准到达标定的中心位置 |
| ![第4张图片数据](https://github.com/evolzed/armlogic/blob/master/docs/pic/Distinguish/collection4.bmp) | 第一个瓶子：646，434，87，60；第二个瓶子：823，438，37，60  | 机械臂吸盘精准到达标定的中心位置 |
---
# 4.**总结**
=======
  * LICENSE.md   

----