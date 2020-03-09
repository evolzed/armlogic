#本文件示例不同的图像分块儿代码 以及建立文件夹的代码
from src.Vision.video import Video
from src.Vision.interface import imageCapture
import cv2


gleft = 0
gtop = 0
gw = 100
gh = 100
work_hours = 0

#九宫格的位置，每次需要根据实际的位置修改下列人名的位置
sudoku = {
          "Hujie":      [gleft,       gleft+gw,    gtop,       gtop+gh,    work_hours],
          "TaoTao":  [gleft+gw,    gleft+2*gw,  gtop,       gtop+gh,    work_hours],
          "LuChenYin":     [gleft+2*gw,  gleft+3*gw,  gtop,       gtop+gh,    work_hours],

          "John":       [gleft,       gleft+gw,    gtop+gh,    gtop+2*gh,  work_hours],
          "FeiFei":     [gleft+gw,    gleft+2*gw,  gtop+gh,    gtop+2*gh,  work_hours],
          "Tina":       [gleft+2*gw,  gleft+3*gw,  gtop+gh,    gtop+2*gh,  work_hours],

          "LouQiGe":    [gleft,       gleft+gw,    gtop+2*gh,  gtop+3*gh,  work_hours],
          "DaPeng":    [gleft+gw,    gleft+2*gw,  gtop+2*gh,  gtop+3*gh,  work_hours],
          "ZhiMing":     [gleft+2*gw,  gleft+3*gw,  gtop+2*gh,  gtop+3*gh,  work_hours]
          }

employee = {
          0: "Hujie",
          2: "Hujie",
          1: "TaoTao",
          3: "LuChenYin",

          4: "John",
          5: "FeiFei",
          6: "Tina",

          7: "LouQiGe",
          8: "DaPeng",
          9: "ZhiMing"
          }


videoDir = "E:\\1\\1.avi"
bgDir = videoDir
if __name__ == '__main__':
    avi = Video(videoDir)
    bgAvi = Video(bgDir)
    imgCapObj = imageCapture(None, avi, bgAvi)

    cv2.namedWindow("window", 0)
    cv2.resizeWindow("window", 1920, 1080)

    while 1:
        curr_cap, nFrame, t = imgCapObj.getImage()
        show = curr_cap.copy()
        for key in sudoku.keys():
            partition = show[sudoku[key][2]:sudoku[key][3], sudoku[key][0]:sudoku[key][1]]
            cv2.imshow(str(key), partition)
        cv2.rectangle(show, (sudoku[key][0], sudoku[key][2]), (sudoku[key][0] + gw, sudoku[key][2] + gh),
                      (0, 255, 255))

        cv2.imshow("window", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 按q退出 并保存excel
            break

