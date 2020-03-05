#本文件计算所有耗时算法（图像算法为主）的运行时间
from timeit import default_timer as timer


class CostTimeCal:
    def __init__(self, CostTimeTypes, isCaled):
        '''

        :param CostTimeTypes:   字符串 说明是计算哪种消耗时间，
        例如'LkTrackTime'，'YoloTime','BglearnTime'
        :param isCaled:  布尔值,说明是否计算
        '''
        self.__CostTimeType = CostTimeTypes   #str 说明是计算哪种消耗时间
        self.__isCal = isCaled
        self.__startTime = 0
        self.__endTime = 0
        self.__costTime = -1

    def calSet(self):
        if self.__isCal is True:
            self.__startTime = timer()

    def calEnd(self):
        if self.__isCal is True:
            self.__endTime = timer()
            costTime = self.__endTime - self.__startTime
            self.__costTime = costTime
        return self.__costTime

    def printCostTime(self):
        if self.__isCal is True:
            print(str(self.__CostTimeType)+" = %f ms" % (self.__costTime * 1000))
        else:
            pass




# def costTimeCalInit():
#
# trackStartTime = 0
# trackEndTime = 0
# if statisticTrackTime is True:
#     trackStartTime = timer()
# p0, label, LKtrackedList = self.imgproc.trackObj(featureimg, secondimg, drawimg, label, p0, deltaT)
# if statisticTrackTime is True:
#     trackEndTime = timer()
# trackCostTime = trackEndTime - trackStartTime
# print("TrackCostTime!!!!!!!!!!!!!!!!!!!!!! = %f ms" % (trackCostTime * 1000))