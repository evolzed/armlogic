import cv2
import numpy as np

class KF():

    # kalman = cv2.KalmanFilter(4, 2)
    # # 设置测量矩阵
    # kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # # 设置转移矩阵
    # kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # # 设置过程噪声协方差矩阵
    # kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    #
    # # frame = np.zeros((960, 1080, 3), np.uint8)
    # # # 初始化测量坐标和target运动预测的数组
    # last_measurement = current_measurement = np.array((2, 1), np.float32)
    #
    # last_prediction = current_prediction = np.zeros((2, 1), np.float32)


    def targetMove(self, x, y):

        # 定义全局变量
        global kalman, frame, current_measurement, last_measurement, current_prediction, last_prediction

        # 初始化
        last_measurement = current_measurement

        last_prediction = current_prediction
        # 传递当前测量坐标值
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

        kalman.correct(current_measurement)

        current_prediction = kalman.predict()

        # 上一次测量值
        lmx, lmy = last_measurement[0], last_measurement[1]

        cmx, cmy = current_measurement[0], current_measurement[1]

        lpx, lpy = last_prediction[0], last_prediction[1]

        cpx, cpy = current_prediction[0], current_prediction[1]

        cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 100, 0))

        cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))

        print(lpx, cmx, cpx, lpy, cmy, cpy)


if __name__ == '__main__':

    x, y = 0, 0
    kalman = cv2.KalmanFilter(4, 2)
    # 设置测量矩阵
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # 设置转移矩阵
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # 设置过程噪声协方差矩阵
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    frame = np.zeros((960, 1080, 3), np.uint8)
    # # 初始化测量坐标和target运动预测的数组
    last_measurement = current_measurement = np.array((2, 1), np.float32)

    last_prediction = current_prediction = np.zeros((2, 1), np.float32)
    kf = KF()
    while True:
        kf.targetMove(x, y)
        cv2.imshow("kalman_tracker", frame)
        # test data
        x += 3
        y += 1
        if (cv2.waitKey(30) & 0xff) == 27:
            break

    cv2.destroyAllWindows()
