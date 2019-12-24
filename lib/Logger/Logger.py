import sys
import time
import datetime

currenttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
currentdatetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class Logger(object):

    def __init__(self, fileN = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    #filter算法，提取需求信息；
    def write(self, message):

        # msg =  currentdatetime + message
        # if (message.find("timecost") != -1):  #假如message内包含字符串"ok"，则写入log，否则不记录
        #     self.terminal.write(msg)
        #     self.log.write(msg)

        msg = currentdatetime + message
        if (message.find("box") and (message.find("timeCost") or message.find("bgTimeCost") or message.find("nFrame")) != -1):  #假如message内包含字符串"box"，则写入log，否则不记录
            self.terminal.write(msg)
            self.log.write(msg)


    def flush(self):
        pass
