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

<table>
	<tr>
	    <th>ModelName</th>
	    <th>Functions/methods</th>
	    <th>Description</th>
        <th>Related Functions</th>
	</tr>
	<tr>
        <td rowspan="5">loadYolo()</td>
	    <td>__init__()</td>
        <td>模型参数初始化(包含model_path、anchors_path、classes_path等)</td>
	    <td>调用generate()方法，初始化boxes，scores， classes</td>
	</tr>
	<tr>
	    <td>_get_class()</td>
        <td>存放类别的.txt文件，返回需要识别的类别列表</td>
	    <td>采用python自带的文件操作方法with open(),返回文件内容列表</td>
	</tr>
	<tr>
	    <td>_get_anchors()</td>
        <td>将anchors转换成np.array,形状是(*, 2)</td>
	    <td>采用python自带的文件操作方法with open()，返回文件内容列表</td>
	</tr>
	<tr>
	    <td>generate()</td>
        <td>返回预测框列表，评分列表，类别列表</td>
	    <td>使用load_model()、yolo_eval()</td>
	</tr>
	<tr><td>detect_image()</td>
        <td>检测输入图像的函数</td>
	    <td>调用letterbox_image():不损坏原图尺寸比例进行填充；PIL下的ImageDraw模块中的Draw()->对图像进行画框标注等操作;</td>
	</tr>
	<tr>
        <td rowspan="6">testRun()</td>
        <td>detect_video()</td>
	    <td>进行实时视频流识别(此方法中集成海康相机驱动进行实时识别)，包括了相机启动自检，启动失败后控制台输出ERROR_MESSAGE并将ERROR写入日志</td>
        <td>HIKvison_camera模块</td>
    </tr>
    <tr>MESSAGE
        <td>connectCam()</td>
        <td>获取检测到的设备编号，连接设备</td>
	    <td>GrabVideo.get_device_num()、GrabVideo.connect_cam()</td>
    </tr>
    <tr>
        <td>grabVideo()</td>
        <td>获取相机的视频流</td>
	    <td>利用封装好的GrabVideo包进行获取</td>
    </tr>
    <tr>
        <td>detect_image()</td>
        <td>将数据流传给yoloCNN</td>
	    <td>cv2.cvtColor()[色彩空间转换]、PIL.Image()[转换成网络需要的imageObject]</td>
    </tr>
     <tr>
        <td>detect_img()</td>
        <td>尝试识别预设的照片，已知图像中的物体以及种类</td>
	    <td>调用detect_image()、PIL.Image模块</td>
    </tr>
    <tr>
        <td>cam.MV_CC_GetOneFrameTimeout()</td>
        <td>使用相机驱动调用视频流中的下一帧图像数据</td>
	    <td>/</td>
    </tr>
    <tr>
        <td rowspan="3">testRun完成后，checkState()[1:init 2：run 3：stop]</td>
        <td>gState==1</td>
	    <td>reInitCNN()</td>
        <td>loadYolo()</td>
    </tr>
	<tr>
        <td>gState==2</td>
	    <td>当检测到系统信号2时，开始运行检测</td>
        <td>detect_video(),创建yolo对象，获取数据流进行识别</td>
    </tr>
	<tr>
	    <td>gState==3</td>
        <td>检测到系统指令3时，停止网络，关闭相机驱动</td>
	    <td>GrabVideo.destroy()[清空保存在内存中的相机数据，销毁相机对象]、yolo.close_session()</td>
	</tr>
</table>


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
