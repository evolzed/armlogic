# BottleSort0.2设计说明书

# Objective：Control using existing libraries to perform simple blast task.

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
![FlowChart](https://github.com/evolzed/armlogic/blob/BottleSort0.1/docs/pic/FlowChart/BS0.1FC.png)

## 2.3 数据变量命名规则

|   类型   | 命名举例 |              描述               | 第一元素 | 第二元素 |  第三元素 |  第四元素 |  第五元素 |
| :------: | :------: | :-----------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| global |  gState  | #1 = 初始阶段, 2 = 运行阶段, 3 = 停止阶段 | int gState |
| library |  airDict  | #User input air pressure nozzle locations | int x | int y | int z | int type |
| library |  bgDict  | #address to image file location and processed flag | #%.JPG | bool processed |
| global |  gDir  | #direction of the moving belt | int gDir | |
| library |  bottleDict  | #sorted bottle information | {"image":imagedata,"box":[(bottletype1, confidence, xmin, ymin, xmax, ymax),(bottletype2, confidence, xmin, ymin, xmax, ymax),···]，"bgTimeCost":time,"timeCost":time,"nFrame":camNumFrame} |  |  | | | Time processed |
| path | bgPic | #file location for bgPic |

## 2.4 功能包文档填写说明

|   Class   | Function |           Description          | Input | Output | Return |
| :------: | :------: | :-----------------------------: | :----: | :----: | :----: |
| Main |  | 主方法，系统流程 | gState |  |  |
| Image |  generate  | 返回预测框列表，评分列表，类别列表, 使用load_model()、yolo_eval() | | | |
| Image |  connectCam  | 获取检测到的设备编号，连接设备GrabVideo.getDeviceNum()、GrabVideo.connectCam() | | | |
| Image |  grabVideo  | 获取相机的视频流,利用封装好的GrabVideo包进行获取 | | %bgPic.JPG | |
| Camera |  getImage  | get a frame of opencv format from camra | %_cam  %_data_buf %_nPayloadSize|  | %frame|
| Camera | getCamFps  | get  fps of camera output in str type  | %nFrameNum  |  | %fps|
| Image |  loadYolo  | Tiny-Yolov3模型参数初始化(包含model_path、anchors_path、classes_path等), 调用generate()方法，初始化boxes，scores， classes | %.py %.pt | | |
| Image |  detectImage  | 检测输入图像的函数, 调用letterbox_image():不损坏原图尺寸比例进行填充；PIL下的ImageDraw模块中的Draw()->对图像进行画框标注, 将数据流传给yoloCNN，cv2.cvtColor()[色彩空间转换]、PIL.Image()[转换成网络需要的imageObject]; | | | |
| Image |  studyBackgroundFromCam  | get 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk| %cam | |  |
| Image |  avgBackground  | learn the backgroud from disk then  accumulate every frame difference,accumulate every frame  | %img |  |  |
| Image |  createModelsfromStats  | average the frame and frame difference to get the background model| %I %dst | bottleDict||
| Image |  backgroundDiff  | use the background to segment the frame pic| %src %dst | ||
| Image |  checkState  | [1:init 2：run 3：stop], 停止网络，关闭相机驱动

----
#  3.**测试BS0.1**
<table><!--此处为注释：<td>要显示的内容需要写在该标签对中</td>-->
    <th colspan="9" align="center">所属项目</th><!--colspan属性表示一行中要合并几列-->
    <tr>
        <th>用例编号</th>
        <th>所属模块</th>
        <th>用例方法</th>
        <th>用例标题</th>
        <th>预期</th>
        <th>实际</th>
        <th>修改建议</th>
        <th>优先级</th>
        <th>时间</th>
	</tr>
	<tr>
		<td rowspan="5"><!--colspan属性表示一列中要合并几行-->
            SXCX-0.1
		</td>
		<td rowspan="5">
            Model
		</td>
		<td rowspan="5">
            性能测试
		</td> 
		<td>内容1
		</td>
	    <td>内容2
		</td>
		<td>内容3
		</td>
		<td>内容4
		</td> 
		<td>内容5
		</td>
	    <td  rowspan="5">time
		</td>
	</tr>
	<tr>
	    <td>内容6
		</td>
		<td>内容7
		</td>
		<td>内容8
		</td> 
		<td>内容9
		</td>
	    <td>内容10
		</td>
	</tr>
	<tr>
	    <td>内容11
		</td>
		<td>内容12
		</td>
		<td>内容13
		</td> 
		<td>内容14
		</td>
	    <td>内容15
		</td>
	</tr>
</table>

----
# 4.**总结**
=======
  * LICENSE.md   
  
----
## 2.2 数据库结构

<table>
	<tr>
	    	<th>类型</th>
	    	<th>名称</th>
	    	<th>介绍</th>
		<th>第一元素</th>
		<th>第二元素</th>
		<th>第三元素</th>
		<th>第四元素</th>
		<th>第五元素</th>
		<th>第六元素</th>
	</tr >
	<tr >
	    	<td rowspan="5">global</td>
	    	<td>gState</td>
	    	<td>#system state, 1 = INIT, 2 = RUN, 3 = STOP </td>
		<td>int gState</td>
	</tr>
	<tr >
	    	<td>gDir</td>
	    	<td>#运动方向的角度 0°~360°  </td>
		<td>int gDir </td>
		<td>Time processed </td>	
	</tr>
	<tr >
	    	<td rowspan="4">Dictionary</td>
	    	<td>airDict</td>
	    	<td>#喷嘴位置坐标  </td>
		<td>int x </td>
		<td>int y </td>	
		<td>int z </td>	
		<td>int type </td>
		<td>int frame </td>
		<td>Time processed </td>
</table>


## 2.3 功能包文档填写说明
<table>
	<tr>
	    <th>Class</th>
	    <th>Functions/methods</th>
	    <th>Description</th>  
	</tr >
	<tr >
	    <td rowspan="9">Control</td>
	    <td>Blast</td>
	    <td>#send a signal to air pressure nozzle upon running code</td>
	</tr>
</table>


#  3.**系统总体设计框架**
## 3.1 系统流程图
![FlowChart](https://github.com/evolzed/armlogic/blob/BottleSort0.1/docs/pic/FlowChart/BS0.2FC.png)
## 3.2 功能包及其实现逻辑
**PC(代替TX2)**

| class |    function    |   description   | 依赖包                                       | 输入参数  | 输出参数  |
| :---: | :------------: | :-------------: | -------------------------------------------- | :-------: | :-------: |
|  PC   | relayService() | #继电器执行模块 | #time<br />#modbus<br />#socket<br />#serial       | gBlasting |     /     |
|  PC   | listenBlast()  |    #监听模块    | #socket<br />#dataBase                       | gDataBase | gBlasting |
|  Image   | getBeltSpeed()  | #get belt speed direction and valu e,pixel per second   |   bottleDict                     |  | beltSpeed |
|  Image   | getBottleDetail()  | #get belt speed direction and valu e,pixel per second   |bottleDict |  | bottleDetail |
|  Image   | getBottleID()  | #get bottle ID by track and beltSpeed   |   bottleDict                     | beltSpeed | bottleID |

**实现逻辑：**

* **PC.relayService()**

  获取PC.listenBlast()发出的gBlasting；

  调用modbus包中的方法处理gBlasting，并调用serial包完成执行

* **PC.listenBlast()**

  监听dataBase的gDataBase；

  ​	gDir转换成的向量与gBottleDict做处理，转换成瓶子在实际方向上的坐标；

  ​	比较上步转换的坐标值与airDict，

  ​	假如在设定范围内（DELTA_X），输出信号gBlasting；
#  4.**测试BS0.2**
| 测试流程 | ---------------描述---------------- |      |
| :------: | :---------------------------------: | ---- |
|   条件   |                                     |      |
|   内容   |                                     |      |
|   结果   |                                     |      |
| 常见错误 |                                     |      |
| 解决办法 |                                     |      |

# 5.**总结**
>>>>>>> 19fe224bc4a3a41c95d01685e60cc606267da219
|         项目          | ---------------描述---------------- |      |
| :-------------------: | :---------------------------------: | ---- |
|     当前方案优点      |                                     |      |
|     当前方案缺点      |                                     |      |
| 改进方案(version 0.2) |                                     |      |
