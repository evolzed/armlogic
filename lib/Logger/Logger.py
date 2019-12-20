import sys
import time
import datetime

currenttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
currentdatetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class Logger(object):

    def __init__(self, fileN = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        msg =  currentdatetime + message
        self.terminal.write(msg)
        #self.log.write(currenttime + ": ")
        self.log.write(msg)

    def flush(self):
        pass
