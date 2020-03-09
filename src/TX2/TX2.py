#! /usr/bin/python
import socket  #socket通信

import Jetson.GPIO as GPIO  # Jetson.GPIO, for controlling GPIOs

import time


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #构造器创建 socket 实例

server_address = ('127.0.0.1', 12345)  #本地server地址，端口号12345

sock.bind(server_address)  #绑定到本机IP地址和端口

sock.listen(1)  #开始监听来自客户端的连接

#pinService

Pin_1 = 13
Pin_2 = 15
Pin_3 = 18
Pin_4 = 22
Pins = [Pin_1,Pin_2,Pin_3,Pin_4]

GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)

GPIO.setup(Pins,GPIO.OUT)

delta_T = 0.02

#logPressure


#listenBlast