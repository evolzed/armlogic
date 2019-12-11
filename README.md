# BottleSort0.1设计说明书

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
	    	<th>Class</th>
		<th>Functions/methods</th>
		<th>Description</th> 
        	<th>Parameter 1</th>
        	<th>Parameter 2</th>
        	<th>Return </th>
	</tr >
	<tr >
	    	<td rowspan="4">Image</td>
	    	<td>cameraOn</td>
	    	<td>功能描述</td>
        	<td></td>
       	 	<td></td>
        	<td>return gCameraGo</td>
	</tr>
	<tr>
	    	<td>getImage</td>
	    	<td>功能描述</td>
        	<td>.raw</td>
        	<td>.JPG</td>
        	<td>return bgDict</td>
	</tr>
	<tr>
	    	<td>bgLearn</td>
	    	<td>功能描述</td>
        	<td>bgDict</td>
        	<td></td>
        	<td>return bgDict</td>
	</tr>
	<tr>
	    	<td>checkImage</td>
	    	<td>功能描述</td>
        	<td>bgDict</td>
        	<td></td>
        	<td>return bottleDict</td>
	</tr>
    	<tr >
	    	<td rowspan="3">Tool</td>
	    	<td>mySQL.createDict</td>
	    	<td>功能描述</td>
        	<td></td>
        	<td></td>
        	<td>return bgDict, bottleDict</td>
	</tr>
	<tr>
	    	<td>mySQL.updateDict</td>
	    	<td>功能描述</td>
        	<td>%FILENAME</td>
        	<td>str</td>
        	<td>return %FILENAME</td>
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
