import numpy as np
import imageio

rawfile = np.fromfile('AfterConvert_RGB.raw', dtype=np.float16)  # 以float32读图片
print(rawfile.shape)
rawfile.shape = (1280, -1)
print(rawfile.shape)
print(rawfile.dtype)
# b = rawfile.astype(np.uint8)  # 变量类型转换，float32转化为int8
# print(b.dtype)
# imageio.imwrite("0.jpg", b)

import matplotlib.pyplot as pyplot

pyplot.imshow(rawfile)
pyplot.show()



# from PIL import Image

# rawData = open("AfterConvert_RGB.raw", 'rb').read()
# # http://www.sharejs.com
# imgSize = (960, 960)
# # Use the PIL raw decoder to read the data.
# # the 'F;16' informs the raw decoder that we are reading
# # a little endian, unsigned integer 16 bit data.
# img = Image.frombytes('L', imgSize, rawData, 'raw', 'F;64')
# img.save("foo.png")


# from PIL import Image
# im = Image.open("AfterConvert_RGB.raw")
# rgb_im = im.convert("RGB")
# rgb_im.save('img.jpg')