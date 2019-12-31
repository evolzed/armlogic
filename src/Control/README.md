# MovePlanning使用说明
# 1.**环境搭建**

## 1.1 硬件环境

* 运行平台： PC/Win10_x86_64

* 机械臂

  + 品牌：UFACTORY 
  + 型号：uArm Swift Pro
  + 接口协议：RS485 

## 1.2 软件环境
* PyCharm 2018.3.5
* python==3.6.5
* numpy==1.16.5
* 标准库：os, sys, math, operator
* uArmSDK
* [GitLab](http://192.168.0.203:8088/armlogic/BottleSort/tree/MovePlanning/src/Control)

## 1.3 通讯接口
* USB to RS485

----
# 2.**定义及规则**

## 2.1 文件夹
* BorrleSort/
  * lib/          #库
  * src/          #源代码
  * README.md     #使用说明技术文档 
----
## 2.2 数据变量命名

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
		<th>第七元素</th>
		<th>第八元素</th>
	</tr >
	<tr >
	    	<td rowspan="5">loacl</td>
	    	<td>bottleInfo</td>
	    	<td>存储了瓶子信息的数组 </td>
		<td>像素坐标u</td>
        <td>像素坐标v </td>
        <td>瓶子姿态角</td>
        <td>瓶子宽度(像素坐标系)</td>
        <td>世界坐标系下坐标Xw</td>
        <td>世界坐标系下坐标Yw</td>
        <td>世界坐标系下坐标Zw</td>
        <td>瓶子到机械臂底座中心的距离</td>
	</tr>
	<tr >
	    	<td>local</td>
	    	<td>RobotOn</td>
		<td>机器人是否连接</td>
		<td>机械臂连接状态(0/1)</td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
	</tr >
	<tr>
	    	<td>local</td>
	    	<td>uArmbottom_P</td>
		<td>机械臂底座中心位置矢量(机械臂基坐标系)</td>
		<td> X 坐标</td>
		<td> Y 坐标</td>
		<td> Z 坐标</td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
	</tr>
	<tr>
	    	<td>local</td>
	    	<td>uArmbottom_R</td>
		<td>世界坐标系绕z轴转90度，与机械臂基坐标系重合，生成旋转矩阵</td>
		<td> 绕Z轴的转角</td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
	</tr>
	<tr>
	    	<td>local</td>
	    	<td>radio</td>
		<td>瓶子宽度的像素值转换到世界坐标系的Z坐标值的比例</td>
		<td> 比例值</td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
	</tr>
	<tr>
	    	<td>local</td>
	    	<td>distance2uArm</td>
		<td>瓶子到机械臂底座中心的距离</td>
		<td> 距离值</td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
		<td> </td>
	</tr>
</table>


#  3.**设计框架**
## 3.1 功能实现流程图
![Flowchart](http://192.168.0.203:8088/armlogic/BottleSort/blob/MovePlanning/docs/pic/FlowChart/MovePlanning.jpg)
## 3.2 功能包及函数


| class |    function    |   description   | 依赖包                                       | 输入参数  | 输出参数  |
| :---: | :------------: | :-------------: | -------------------------------------------- | :-------: | :-------: |
| swift_api | waiting_ready  | # Waiting the uArm ready   |   Swift                     | None | None |
| swift_api | send_cmd_async | # Send cmd async cmd |SwiftAPI | msg: cmd | None |
| swift_api | set_digital_direction | # Set digital direction | SwiftAPI |PinID, Value | None |
| swift_api | set_acceleration  | # Set the acceleration, only support firmware version > 4.0   |   SwiftAPI                     | Value| None |
| swift_api| set_position|# Set the robot position|SwiftAPI|X,Y,Z,speed,wait,cmd|None|
| swift_api|flush_cmd|# Wait until all async command return or timeout |SwiftAPI|None|None|
| swift_api|set_wrist|# Set the wrist angle|SwiftAPI|角度|None|
| swift_api|get_servo_angle| # Get the servo angle|SwiftAPI|servo_id(0~3)|角度|
| swift_api|set_digital_output|# Set digital output value|SwiftAPI|PinID, value, wait, timeout||
|numpy|shape|# 求数组行数|None|数组|行数|
|numpy|array|# 创建数组|None|数组元素|数组|
|numpy|dot|# 求矩阵的乘积|None|矩阵1，矩阵2|结果|
|numpy|sqrt|# 开平方|None|待求表达式|结果|
|operator|itemgetter| # Return a callable object that fetches the given item(s) from its operand.|None|row|key|

