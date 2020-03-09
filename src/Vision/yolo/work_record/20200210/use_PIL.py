from PIL import Image


src = Image.open("images/bottle_01.png")
# src = cv2.imread("images/12.jpg")
dst = Image.open("images/bg.jpg")
# dst = cv2.imread("images/bg.jpg")
dst.paste(src, (100, 200))
dst.show()