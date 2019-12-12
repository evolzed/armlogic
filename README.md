# BottleSort0.1设计说明书

# 1.**环境搭建**

## 1.1 硬件环境

* 运行平台： PC/Win10_x86_64**&**Linux

* 工业相机

  + 品牌：HIKVision 
  + 接口协议：Gigabit Ethernet（GigE协议）
  + 型号:MV-CE013-50GC
  + **注意：相机完成了Win10下驱动调用，Linux平台未测试**
  
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

----

# 2.**定义及规则**

## 2.1 文件夹
* project_root/
  * lib/          #库
  * docs/         #技术文档
  * src/          #源代码
  * test/         #测试
  * README.md     
  * LICENSE.md     
  
## 2.2 系统流程图
![FlowChart](https://github.com/evolzed/armlogic/blob/BottleSort0.1/docs/pic/FlowChart/BS0.1FC.png)

## 2.3 数据变量命名规则

|   类型   | 命名举例 |              描述               | 第一元素 | 第二元素 |  第三元素 |  第四元素 |  第五元素 | 
| :------: | :------: | :-----------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| global |  gState  | #1 = 初始阶段, 2 = 运行阶段, 3 = 停止阶段 | int gState |
| library |  airDict  | #User input air pressure nozzle locations | int x | int y | int z | int type |
| library |  bgDict  | #address to image file location and processed flag | #%.JPG | bool processed |
| global |  gDir  | #direction of the moving belt | int gDir |
| library |  bottleDict  | #recognised bottle information | int x | int y | int z | int type | float frame | Time processed |

## 2.4 功能包文档填写说明

|   Class   | Function |           Description          | Input | Output | Return | 
| :------: | :------: | :-----------------------------: | :----: | :----: | :----: | 
| Image |  _get_class  | 存放类别的.txt文件，返回需要识别的类别列表采用python自带的文件操作方法with open(),返回文件内容列表 | | | | |
| Image |  _get_anchors  | 将anchors转换成np.array,形状是(*, 2), 采用python自带的文件操作方法with open()，返回文件内容列表 | | | | |
| Image |  bgLearn  | learn the backgroud by pics from cam then get a background model |capture |background model |bgDict | |
| Image |  generate  | 返回预测框列表，评分列表，类别列表, 使用load_model()、yolo_eval() | | | | |
| Image |  loadYolo  | 模型参数初始化(包含model_path、anchors_path、classes_path等), 调用generate()方法，初始化boxes，scores， classes | input | | | | |
| Image |  detect_image  | 检测输入图像的函数, 调用letterbox_image():不损坏原图尺寸比例进行填充；PIL下的ImageDraw模块中的Draw()->对图像进行画框标注, 将数据流传给yoloCNN，cv2.cvtColor()[色彩空间转换]、PIL.Image()[转换成网络需要的imageObject]; | | | | |
| Image |  connectCam  | 获取检测到的设备编号，连接设备GrabVideo.get_device_num()、GrabVideo.connect_cam() | | | | |
| Image |  grabVideo  | 获取相机的视频流,利用封装好的GrabVideo包进行获取 | | | | |
| Image |  cam.MV_CC_GetOneFrameTimeout()  | 获取相机的视频流,利用封装好的GrabVideo包进行获取 ，使用相机驱动调用视频流中的下一帧图像数据| | | | |
| Image |  checkImage  | check the cnn detected result by image process and image track and then update the bottle dict|capture | bottledict| | |
| Image |  checkState  | [1:init 2：run 3：stop], 停止网络，关闭相机驱动</td>
	    <td>GrabVideo.destroy()[清空保存在内存中的相机数据，销毁相机对象]、yolo.close_session() | | | | |


----
#  3.**测试BS0.1**
| 测试流程 | ---------------描述---------------- |      |
| :------: | :---------------------------------: | ---- |
|   条件   |                                     |      |
|   内容   |                                     |      |
|   结果   |                                     |      |
| 常见错误 |                                     |      |
| 解决办法 |                                     |      |

----
# 4.**总结**
|         项目          | ---------------描述---------------- |      |
| :-------------------: | :---------------------------------: | ---- |
|     当前方案优点      |                                     |      |
|     当前方案缺点      |                                     |      |
| 改进方案(version 0.2) |                                     |      |
