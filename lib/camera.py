# -- coding: utf-8 --
#!/bin/python
import sys
import threading
import msvcrt
import cv2
import numpy as np
from PIL import Image

from ctypes import *
# sys.path.append("HikMvImport")
from lib.HikMvImport.MvCameraControl_class import *

g_bExit = False
# 设置保存的帧间隔
per_frame = 2
# 状态常量 ERR:-1 OK:0
OK = 0
ERR = -1


np.set_printoptions(threshold=np.inf)  # 设置numpy数组完全显示

deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE


class Camera(object):
    """海康相机类"""
    # 为线程定义一个函数
    def work_thread(self, cam=0, pData=0, nDataSize=0):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        n = 0
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
        # out = cv2.VideoWriter("out.avi", fourcc, 30.0, (1280, 960))
        while True:
            temp = np.asarray(pData)
            temp = temp.reshape((960, 1280, 3))
            # print(temp)
            # print(temp.shape)
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

            ret = cam.MV_CC_GetOneFrameTimeout(byref(pData), nDataSize, stFrameInfo, 1000)
            # 利用PIL进行图片显示，注意位置
            # image = Image.frombytes("RGB", (stFrameInfo.nWidth, stFrameInfo.nHeight), pData)
            # image.show("test")
            n += 1
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            else:
                print("no data[0x%x]" % ret)
            if g_bExit == True:
                break

    def get_device_num(self):
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR, ERR
        if deviceList.nDeviceNum == 0:
            print("相机初始化失败！|find no device!Make sure you have connected your camera via netline")
            # sys.exit()
            return ERR, ERR, ERR
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

    def connect_cam(self, nConnectionNum):
        """
        :param nConnectionNum: 选择检测到得相机序号,默认是0
        :return: cam实例对象， 数据流data_buf
        """
        print("Default use the first device found！")
        if int(nConnectionNum) >= deviceList.nDeviceNum:
            print("intput error!")
            # sys.exit()
            return ERR, ERR, ERR
        # ch:创建相机实例 | en:Creat Camera Object
        cam = MvCamera()

        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR, ERR

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        # print(ret)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            # sys.exit()
            # error code -1 初始化失败
            return ERR, ERR, ERR
        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR, ERR

        # ch:获取数据包大小 | en:Get payload size
        stParam = MVCC_INTVALUE()
        # stParam --> nCurValue', 'nInc', 'nMax', 'nMin', 'nReserved
        # print(stParam.nInc)
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

        ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR, ERR
        nPayloadSize = stParam.nCurValue

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            # sys.exit()
            return ERR, ERR, ERR
        data_buf = (c_ubyte * nPayloadSize)()
        return cam, data_buf, nPayloadSize


    def destroy(self, _cam, _data_buf):
        # ch:停止取流 | en:Stop grab image
        ret = _cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            del _data_buf
            # sys.exit()
            return ERR, ERR, ERR

        # ch:关闭设备 | Close device
        ret = _cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close deivce fail! ret[0x%x]" % ret)
            del _data_buf
            # sys.exit()
            return ERR, ERR, ERR
        # ch:销毁句柄 | Destroy handle
        ret = _cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            del _data_buf
            # sys.exit()
            return ERR, ERR, ERR
        del _data_buf


def main():
    """主程序"""
    cam = Camera()
    nConnectionNum = cam.get_device_num()
    # cam.get_device_num()
    _cam, _data_buf, _nPayloadSize = cam.connect_cam(nConnectionNum)

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
        hThreadHandle = threading.Thread(target=cam.work_thread, args=(_cam, _data_buf, _nPayloadSize))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    print("press a key to stop grabbing.")
    msvcrt.getch()
    global g_bExit
    g_bExit = True
    hThreadHandle.join()
    cam.destroy(_cam, _data_buf)


if __name__ == "__main__":
    main()
