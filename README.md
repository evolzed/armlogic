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
## 2.1 数据变量命名规则
|   类型   | 命名举例 |              描述               |
| :------: | :------: | :-----------------------------: |
| 全局变量 |  gState  | 1初始阶段, 2运行阶段, 3停止阶段 |

## 2.2 功能包文档填写说明
<table>
	<tr>
	    <th>Class</th>
	    <th>Functions/methods</th>
	    <th>Description</th>  
	</tr >
	<tr >
	    <td rowspan="9">type</td>
	    <td>fun1</td>
	    <td>功能描述</td>
	</tr>
	<tr>
	    <td>fun2</td>
	    <td>功能描述</td>
	</tr>
	<tr>
	    <td>fun3</td>
	    <td>功能描述</td>
	</tr>
	<tr>
	    <td>fun4</td>
	    <td>功能描述</td>
	</tr>
	<tr><td>fun5</td>
	    <td>功能描述</td>
	</tr>
	<tr>
	    <td>fun6/td>
	    <td>功能描述/td>
	<tr>
	    <td>fun7</td>
	    <td>功能描述</td>
	</tr>
	<tr>
	    <td>fun8</td>
	    <td>功能描述</td>
	</tr>
	<tr>
	    <td >fun9</td>
	    <td>功能描述</td>
	</tr>
	<tr>
	    <td >name</td>
	    <td>用户自定义</td>
	    <td>控件名称</td>
	</tr>
	<tr>
	    <td >value</td>
	    <td >用户自定义</td>
	    <td >默认文本值</td>
	</tr>
	<tr>
	    <td >size</td>
	    <td >正整数</td>
	    <td >控件在页面中的显示宽度</td>
	</tr>
	<tr>
	    <td >checked</td>
	    <td >checked</td>
	    <td >定义选择控件默认被选中项</td>
	</tr>
	<tr>
	    <td >maxlength</td>
	    <td >正整数</td>
	    <td >控件允许输入的最多字符</td>
	</tr>
</table>




| 功能包名称 |      className       |    /     |
| :--------: | :------------------: | :------: |
|    功能    | 该功能包所实现的功能 |    /     |
|   依赖包   |      依赖包名称      |    /     |
|  输入参数  |       数据类型       | 变量描述 |
|  输出参数  |       数据类型       | 变量描述 |

| 子函数名称 | functionName |    /     |
| :--------: | :----------: | :------: |
|    功能    |  子函数功能  |    /     |
|   依赖包   |  依赖包名称  |    /     |
|  输入参数  |   数据类型   | 变量描述 |
|  输出参数  |   数据类型   | 变量描述 |

## 2.3 功能包文档
|  class   |               Image                |            描述            |
| :------: | :--------------------------------: | :------------------------: |
|   功能   | Image processing related functions |             /              |
|  依赖包  |                                    |             /              |
| 输入参数 |                .raw                | input from camera at 30fps |
| 输入参数 |                list                |        controlDict         |
| 输出参数 |                list                |           bgDict           |
| 输出参数 |                list                |         bottleDict         |

| function |               Image                | 描述 |
| :------: | :--------------------------------: | :--: |
|   功能   | Image processing related functions |  /   |
|  依赖包  |            class Image             |  /   |
| 输入参数 |                                    |      |
| 输入参数 |                                    |      |
| 输出参数 |                                    |      |
| 输出参数 |                                    |      |

## 2.4 文件夹
* project_root/
  * lib/          #库
  * docs/         #技术文档
  * src/          #源代码
  * test/         #测试
  * README.md     
  * LICENSE.md     

#  3.**系统总体设计框架**
## 3.1 系统流程图
![FlowChart](https://github.com/evolzed/armlogic/blob/BottleSort0.1/docs/pic/FlowChart/BS0.1FC.png)
## 3.2 功能包及其实现逻辑
* 相机
  * 初始化相机(cameraConfig,cameraOn)

        填写规则见上文。
  * 获取图像(getImage)

        填写规则见上文。
* 图像处理与识别(ImageProcess)
  * 背景学习(bgLearn)

        填写规则见上文。
  * 瓶子位置(imageCheck)

        填写规则见上文。
* 数据库搭建(dataBase)

  * 数据库结构

|   名称    |      第1~3个元素      | 第4个元素 |      第5个元素      |
| :-------: | :-------------------: | :-------: | :-----------------: |
| gdataBase | 瓶子的位置坐标(x,y,z) | 拍照时刻t | 瓶子运动速度估计值v |

    举例:gdataBase = [100,200,300,5.0,0.8].

  * 数据写入(updateDatabase)

        填写规则见上文。
#  4.**测试BS0.1**
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
