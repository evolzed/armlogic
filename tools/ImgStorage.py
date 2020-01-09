#! /usr/bin/python3

import pymysql
import paramiko
import time
import datetime
import cv2

class Connect(object):
    def ssh(self, host, port, user, passwd, cmd):
        """
        实现PC远程连接TX2控制，完成对TX2的命令发送指令
        :param host: 远程设备IP地址
        :param port: 远程协议端口，这边通常使用ssh协议端口，端口号为22
        :param user: 远程设备用户名
        :param passwd: 远程设备用户名密码
        :param cmd: 需要对远程设备操作的指令
        :return: 返回了命令的返回值
        """
        client = paramiko.SSHClient()  # 允许连接不在know_hosts文件中的主机
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 连接服务器
        try:
            client.connect(hostname=host, port=port, username=user, password=passwd)  # 执行命令，远程设备的信息
        except:
            print('connect {}\tip fail'.format(host))
        else:
            # print('connect{}\tip successfully'.format(host))
            stdin, stdout, stderr = client.exec_command(command=cmd)  # 对远程设备输出的指令
            result = stdout.read().decode('utf-8')
            # print(result)
            return result
        finally:
            client.close()

class MysqlConnect(object):

    def connectMysql(self):
        """
        连接数据库
        :return: None
        """
        db = pymysql.connect(host='192.168.0.203', user='root', password='rootroot', database='mysql', port=3306)   # 远程连接数据库
        cursor = db.cursor()
        check = cursor.execute("show tables like '{}'".format())
        if check == 0:
            create = cursor.execute(
                "CREATE TABLE {}(Species VARCHAR(20),Brand VARCHAR(20), B_type VARCHAR(20) Iden_time VARCHAR(20), B_status VARCHAR(20), UP_time VARCHAR(20), Pic_name VARCHAR(20), Pic_dir VARCHAR(20), str1 VARCHAR(20), str2 VARCHAR(20), str3 VARCHAR(20))".format())
        addDate = cursor.execute("insert into {}(Species,Brand,B_type,Iden_time,B_status,UP_time,Pic_name,Pic_dir) values('{}','{}','{}',Now(),'{}','{}','{}','{}',)".format())
        cursor.close()


if __name__ == "__main__":
    conn = Connect()
    data = conn.ssh(host="192.168.0.203", port=22, user="armlogic", passwd="admin", cmd="vmstat -w -w")
    data = data.split('\n')[-2].split()
    mem, CPU = data[3], int(data[-5])+int(data[-4])
    # print(mem, CPU)
    '''
    print(data, mem, CPU)
    if mem > 5000000 & CPU < 60:
        for
    else:
        break()
    '''
    capturefile = conn.ssh(host="192.168.0.203", port=22, user="armlogic", passwd="admin", cmd="ls -l /home/armlogic/caspar/test/*.txt")
    for capture in list(capturefile.split('\n')):
        if not len(capture) == 0:
            filedir = capture.split()[-1]
            filename = filedir.split("/")[-1]
            date, dTime, pic = filename.split("_")[0], filename.split("_")[1], filename.split("_")[2]
            print(filename, date, dTime, pic)
            Mconn = MysqlConnect().connectMysql()
            scpFile = conn.ssh(host="192.168.0.203", port=22, user="armlogic", passwd="admin", cmd="scp {} armlogic@192.168.0.203:/home/armlogic/data && mv {} /home/armlogic/caspar/test/move".format(filedir, filedir))
            time.sleep(0.5)
    print("Not File to deal with")

