## acquisition

### 实现前提：

##### 要求：

​	调用工业相机拍摄塑料瓶采集数据。

##### 塑料瓶标准：

​	塑料瓶垂直角度各种姿态的呈现。

##### 工业相机标准：


​	每秒30帧率，每2帧率采集一张照片。

##### 拍摄距离：

​	相机距离塑料瓶120cm。

##### 拍摄时间：

​	根据需求决定拍摄时间。



### 实现过程：

#### 需要的环境：

* win10 X86-64
* python==3.6.5
* opencv-python==3.4.2.16
* numpy==1.16.5




#### 安装环境可能遇到的问题

问题一：win10安装python==3.6.5后,pip升级报错的解决办法

​	1.在Python\Lib\site-packages目录下删除原版本pip文件夹
​	2.以管理员身份运行cmd
​	3.键入python -m ensurepip命令
​	4.键入python -m pip install --upgrade pip命令
​	等待下载即可.。。。。

问题二：安装过程中失败提示“PermissionError: [WinError 5] 拒绝访问”

​	1.找到Python的安装文件夹，点击右键->属性，出现如下图所示对话框

​	2.单击对话框中的“编辑”按钮，选中“Users(LAPTOP-ESE2JG7K\Users)”，将“完全控制”和“修改”两个权限打“√”，出现如下图所示对话框，接着点击“确定”按钮



#### 需要的代码：

​	[https://github.com/evolzed/armlogic/tree/ytt](https://github.com/evolzed/armlogic/tree/ytt)



#### 需要安装的SDK

​	1. ~/Save_JPG/海康SDK/机器视觉工业相机SDK V3.2.0版本Runtime组件包

​	2. 双击压缩包打开，然后点击MVS_SDK_V3_2_0_VC90_Runtime_190626.exe进行安装




#### 运行的命令：

* $ cd Save_JPG/GrabVideo
* $ python Jpg_to_local.py
* 成功后根据提示输入数字 0




​	







