# BottleSort0.2设计说明书

# Objective：Track Function & Optimize Vision Efficiency

# 1.**环境搭建**

## 1.1 硬件环境

* 运行平台： PC/Win10_x86_64**&**Linux

* 工业相机

  + 品牌：HIKVision 
  + 型号：MV-CE013-50GC
  + 接口协议：Gigabit Ethernet（GigE协议）

## 1.2 软件环境
* Ubuntu 18.04
* PyCharm 2018.3.5
* python==3.6.5
* opencv==4.4.1
* tensorflow-gpu==1.6.0或tensorflow-gpu==1.9.0（gpu版本需要配置cuda）
  - Tips：安装cuda：查看自己的gpu是否支持cuda[点击查看我的GPU是否支持安装cuda](https://developer.nvidia.com/cuda-gpus)，如果不支持，只能使用cpu版本的tensorflow，同样安装tensorflow==1.6.0或tensorflow==1.9.0
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
![Flowchart](http://192.168.0.203:8088/armlogic/BottleSort/blob/bottlesort0.2/docs/pic/FlowChart/BS0.2FC.png)
## 3.2 功能包及其实现逻辑


| class |    function    |   description   | 依赖包                                       | 输入参数  | 输出参数  |
| :---: | :------------: | :-------------: | -------------------------------------------- | :-------: | :-------: |
| Vision | getBeltSpeed()  | #get belt speed direction and value,pixel per second   |   bottleDict                     |  | beltSpeed |
| Vision | getBottlePos() | #get Bottle Detail info,include bottle rotate angle and the diameter of bottle |bottleDict |  |  |
| Vision | getBottleTrackID() | #get bottle ID by track|  | | trackDict |
| Vision | getBottleID()  | #get bottle ID by track and beltSpeed   |   bottleDict                     | beltSpeed | bottleID |

* 实现逻辑：

  ...

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
