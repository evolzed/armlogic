# MovePlanning使用说明


# 1.**环境搭建**

## 1.1 硬件环境

* 运行平台： PC/Win10_x86_64**&**Linux

* 机械臂

  + 品牌：UFACTORY 
  + 型号：uArm Swift Pro
  + 接口协议：RS485 

## 1.2 软件环境
* PyCharm 2018.3.5
* python==3.6.5
* numpy==1.16.5
* 标准库：os, sys, math, operator
* [GitLab](http://192.168.0.203:8088/armlogic/BottleSort/tree/MovePlanning/src/Control)

## 1.3 通讯接口
* USB to RS485

----
# 2.**定义及规则**

## 2.1 文件夹
* BorrleSort/
  * lib/          #库
  * src/          #源代码
  * test/         #测试
  * README.md     #使用说明技术文档
  * LICENSE.md   
  
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
	</tr >
	<tr >
	    	<td rowspan="5">global</td>
	    	<td>gState</td>
	    	<td>#system state, 1 = INIT, 2 = RUN, 3 = STOP </td>
		<td>int gState</td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
	</tr>
	<tr >
	    	<td>gDir</td>
	    	<td>#运动方向的角度 0°~360°  </td>
		<td>int gDir </td>
		<td>Time processed </td>
        <td> </td>
        <td> </td>
        <td> </td>
        <td> </td>
	</tr></table>


#  3.**系统总体设计框架**
## 3.1 系统流程图
![Flowchart](http://192.168.0.203:8088/armlogic/BottleSort/blob/MovePlanning/docs/pic/FlowChart/MovePlanning.jpg)
## 3.2 功能包及其实现逻辑


| class |    function    |   description   | 依赖包                                       | 输入参数  | 输出参数  |
| :---: | :------------: | :-------------: | -------------------------------------------- | :-------: | :-------: |
| MovePlanning | getBeltSpeed()  | #get belt speed direction and value,pixel per second   |   bottleDict                     |  | beltSpeed |
| MovePlanning | getBottleAngle() | #get bottle angle (in contrast to belt direction) |bottleDict |  | bottleAngle |
| MovePlanning | getBottleDiameter() | #get bottle diameter | bottleDict | | bottleDiameter |
| MovePlanning | getBottleID()  | #get bottle ID by track and beltSpeed   |   bottleDict                     | beltSpeed | bottleID |


# 4.**总结**
|         项目          | ---------------描述---------------- |      |
| :-------------------: | :---------------------------------: | ---- |
|     当前方案优点      |                                     |      |
|     当前方案缺点      |                                     |      |
| 改进方案(version 0.4) |                                     |      |
