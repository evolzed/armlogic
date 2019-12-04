# 环境要求

1、win10 X86-64

2、python==3.6.5

3、opencv-python==3.4.2.16

4、numpy==1.16.5

5、海康官网提供SDK：<https://www.hikrobotics.com/service/soft.htm?type=2>

# 操作

0、保证相机连接成功，利用SDK中的MVS V3.1.0设置相机ip地址（办公区已配置192.168.0.88）

1、cd 到Save_JPG/GrabVideo/

2、运行python Jpg_to_local.py文件

（PS：如果报错“OSError: [WinError 126] 找不到指定的模块。”提示找不到dll文件，尝试安装海康SDK中的Runtime组件包，不行的话，重启计算机，重新运行python Jpg_to_local.py）

3、选择默认设备0

4、照片 保存在DataSet/文件下

（注意：需要修改，重复运行GrabVideo.py的话，同名文件将被覆盖，可以利用时间戳来命名文件，保证每次的文件名都不一样）