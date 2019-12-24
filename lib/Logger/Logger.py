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
        self.terminal.write(msg)
        if (message.find("bottle") and (message.find("timeCost")) != -1):  #假如message内包含字符串"bottle"以及"timeCost"，则写入log，否则不记录
            self.log.write(msg)


    def flush(self):
        pass
