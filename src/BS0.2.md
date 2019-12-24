# BottleSort0.2设计说明书

# Objective：Control using existing libraries to perform simple blast task.

# 1.**环境搭建**

## 1.1 硬件环境

* 运行平台： PC/Win10_x86_64**&**Linux

* 工业相机

  + 品牌：HIKVision 
  + 型号:MV-CE013-50GC
  + 接口协议：Gigabit Ethernet（GigE协议）
  + **注意：相机完成了Win10下驱动调用，Linux平台未测试**

## 1.2 软件环境
* Ubuntu 18.04
* PyCharm 2018.3.5
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
|         项目          | ---------------描述---------------- |      |
| :-------------------: | :---------------------------------: | ---- |
|     当前方案优点      |                                     |      |
|     当前方案缺点      |                                     |      |
| 改进方案(version 0.2) |                                     |      |
