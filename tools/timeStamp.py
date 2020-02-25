#本文件用于添加各种时间戳
import time
from timeit import default_timer as timer

#年月日时分秒
def getTimeStamp():
    time_day = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    time_day =str(time_day)
    return time_day

if __name__ =="__main__":
    print(getTimeStamp())
