#! /usr/bin/python
import socket
import Jetson.GPIO as GPIO
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('192.168.0.15', 12345)
print "Starting up on %s:%s" % server_address
sock.bind(server_address)
sock.listen(1)

print "Waiting for a connection"
connection, client_address = sock.accept()

Pin_1 = 13
Pin_2 = 15
Pin_3 = 18
Pin_4 = 22
Pins = [Pin_1,Pin_2,Pin_3,Pin_4]

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(Pins,GPIO.OUT)
delta_T = 0.02

while True:

    data = connection.recv(1024)
    print "Receive '%s'" % data

    arr = []
    k = 0
    for i in data:
        #print(Pins[k])
        if (int(i)):
            arr.append(Pins[k])
        #else:
            #arr.append(0)
        k = k + 1
    print(arr)

    Pin = int(data)
    GPIO.output(arr,GPIO.HIGH)
    time.sleep(delta_T)
    GPIO.output(arr,GPIO.LOW)

GPIO.cleanup()
#connection.close()
