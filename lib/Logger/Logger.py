import sys
import time

currenttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

class Logger(object):

    def __init__(self, fileN = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(currenttime + ": ")
        self.log.write(message)

    def flush(self):
        pass
