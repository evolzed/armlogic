# -- coding: utf-8 --
#!/bin/python
import sys
import os
import threading
import time

import cv2
import numpy as np
sys.path.append(os.path.abspath("../../"))
from ctypes import *
# print(sys.path)
# 获取系统架构
import platform
sysArc = platform.uname()
if sysArc[0] == "Windows":
    import msvcrt
    from lib.HikMvImport_Win.MvCameraControl_class import *
elif sysArc[0] == "Linux" and sysArc[-1] == "aarch64":
    import termios
    from lib.HikMvImport_TX2.MvCameraControl_class import *
else:
    print("不支持的系统架构，仅支持win10_64 和 Ubuntu16.04 ARM aarch64！！")
    sys.exit()
from timeit import default_timer as timer

g_bExit = False
# 设置保存的帧间隔
per_frame = 2
# 错误信息
ERR = -1

deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE


class Camera(object):
    """海康相机类"""

    def __init__(self):
        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))
        self.nConnectionNum = self.getDeviceNum()
        print("相机初始化···")
        self.cam = MvCamera()
        self._data_buf, self._nPayloadSize = self.connectCam()
        self.nFrameNumPreOneSec=0.0
        self.prev_time = 0.0
        self.accum_time = 0.0
        self.curr_fps = 0.0
        self.fps = "FPS: ??"
        self.fpsnum = 0.0
        #self.curr_time = 0
        self.curr_time = timer()
        exec_time = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        self.accum_time = self.accum_time + exec_time
        if self.accum_time > 1.0:  # time exceed 1 sec and we will update values
            self.accum_time = 0.0  # back to origin
        if self._data_buf == -1:
            print("相机初始化失败！")
            sys.exit()

    def work_thread(self):
        """为线程定义一个函数"""
        pData = self._data_buf
        nDataSize = self._nPayloadSize
        n = 0
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
        # out = cv2.VideoWriter("out.avi", fourcc, 30.0, (1280, 960))
        while True:
            temp = np.asarray(pData)
            temp = temp.reshape((960, 1280, 3))
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            # 利用opencv进行抽帧采集数据
            # if stFrameInfo.nFrameNum % per_frame == 1:
            #     cv2.imwrite("DataSet/" + str(n) + ".jpg", temp)
            #     print("已保存{}张图片".format((n+1)/per_frame))
            cv2.namedWindow("ytt", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ytt", temp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 利用opencv进行视频保存
            # temp = cv2.flip(temp, 0)
            # out.write(temp)
            # print("视频保存中。。。")
            # cv2.imshow("frame", temp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            ret = self.cam.MV_CC_GetOneFrameTimeout(byref(pData), nDataSize, self.stFrameInfo, 1000)
            # 利用PIL进行图片显示，注意位置
            # image = Vision.frombytes("RGB", (stFrameInfo.nWidth, stFrameInfo.nHeight), pData)
            # image.show("test")
            n += 1
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    self.stFrameInfo.nWidth, self.stFrameInfo.nHeight, self.stFrameInfo.nFrameNum))
            else:
                print("no data[0x%x]" % ret)
            if g_bExit == True:
                break

    def getDeviceNum(self):
        """
        ch:枚举设备 | en:Enum device

        :param: None
        :return: 返回检测到的设备号 如果错误 返回None
        """
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()
        if deviceList.nDeviceNum == 0:
            print("相机初始化失败！|find no device!Make sure you have connected your camera via netline")
            sys.exit()
        print("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                return i
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)
        # return input("please input the number of the device to connect:")

    def connectCam(self):
        """
        连接相机并返回数据

        :return: (data_buf:数据流, nPayloadSize：数据流尺寸) | 错误返回
        """
        print("Default use the first device found！")
        if int(self.nConnectionNum) >= deviceList.nDeviceNum:
            print("intput error!")
            # sys.exit()
            return ERR, ERR
        # ch:创建相机实例 | en:Creat Camera Object
        # cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(self.nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR

        # ch:打开设备 | en:Open device
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        # print(ret)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            # sys.exit()
            # error code -1 初始化失败
            return ERR, ERR
        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR

        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        # stParam --> nCurValue', 'nInc', 'nMax', 'nMin', 'nReserved
        # print(stParam.nInc)
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR
        nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR
        data_buf = (c_ubyte * nPayloadSize)()
        return data_buf, nPayloadSize

    def getImage(self):
        """
        获得图像信息

        :return: (frame:图片信息, nframe:帧号, t:该帧的时间)
        """
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self._data_buf), self._nPayloadSize, self.stFrameInfo, 1000)
        t = time.time()  # 获取当前帧的时间
        if ret == 0:
            pass
            # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            #     self.stFrameInfo.nWidth, self.stFrameInfo.nHeight, self.stFrameInfo.nFrameNum))
        else:
            print("no data[0x%x]" % ret)
        frame = np.asarray(self._data_buf)
        frame = frame.reshape((960, 1280, 3))

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow("rrr", frame)
        # cv2.waitKey(10)
        return frame, self.stFrameInfo.nFrameNum, t

    def destroy(self):
        """
        关闭相机后，删除缓存数据，不然相机需要等待2分钟才能再次使用

        :return: None
        """
        _data_buf = self._data_buf
        # ch:停止取流 | en:Stop grab image
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            del _data_buf
            sys.exit()

        # ch:关闭设备 | Close device
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close deivce fail! ret[0x%x]" % ret)
            del _data_buf
            sys.exit()
        # ch:销毁句柄 | Destroy handle
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            del _data_buf
            sys.exit()
        del _data_buf


    """
    def getFpsInit(self):
        self.prev_time = timer()
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.fpsnum = 0
    """


    def getCamFps(self,nFrameNum):
        """
        get the frame quantity per second

        :param nFrameNum: current frame number
        :return: fps with str type
        """
        self.curr_time = timer()
        exec_time = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps = self.curr_fps+ 1
        if self.accum_time > 1.0:  # time exceed 1 sec and we will update values
            self.fpsnum = nFrameNum - self.nFrameNumPreOneSec
            # print("fpsnum", self.fpsnum)
            self.fps = "FPS: " + str(self.fpsnum)
            self.nFrameNumPreOneSec = nFrameNum  # update the nFrameNumPreOneSec every 1 second
            self.accum_time = 0.0  # back to origin
        return str(self.fps)

    def press_any_key_exit(self):
        """
        主要用于ubuntu系统下按任意键退出相机

        :return: None
        """
        fd = sys.stdin.fileno()
        old_ttyinfo = termios.tcgetattr(fd)
        new_ttyinfo = old_ttyinfo[:]
        new_ttyinfo[3] &= ~termios.ICANON
        new_ttyinfo[3] &= ~termios.ECHO
        #sys.stdout.write(msg)
        #sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
        try:
                os.read(fd, 7)
        except:
                pass
        finally:
                termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)


def main():
    """
    调试使用主函数入口

    :return: None
    """
    cam = Camera()
    # nConnectionNum = cam.getDeviceNum()
    # cam.getDeviceNum()
    # _data_buf, _nPayloadSize = cam.connectCam(nConnectionNum)

    # work_thread(_cam, _data_buf, _nPayloadSize)

    # stFrameInfo = MV_FRAME_OUT_INFO_EX()
    # memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    #
    # while True:
    #     temp = np.asarray(_data_buf)
    #     temp = np.reshape(temp, (960, 1280,3))
    #     print(temp)
    #     # temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    #     cv2.namedWindow("ytt", cv2.WINDOW_NORMAL)
    #     cv2.imshow("ytt", temp)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    try:
        hThreadHandle = threading.Thread(target=cam.work_thread)
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    print("press a key to stop grabbing.")
    if sysArc[0] == "Windows":
        msvcrt.getch()
    elif sysArc[0] == "Linux" and sysArc[-1] == "aarch64":
        cam.press_any_key_exit()
    global g_bExit
    g_bExit = True
    hThreadHandle.join()
    cam.destroy()


if __name__ == "__main__":
    main()
