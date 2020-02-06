import os
base_path = os.path.abspath(".")


def merge_anno(output_path="output/YOLO_darknet", img_path="input"):
    """
    合并单个的txt到annotation

    :param output_path: 需要合并的txt文件保存的文件夹路径
    :param img_path: 需要训练的图片路径
    :return: None
    """
    output_path = os.path.join(base_path, output_path)
    file_list = os.listdir(output_path)
    # annotation文件保存到output同级目录下
    dst_path = os.path.join(os.path.dirname(output_path), "bottle_annotation.txt")
    index = 1
    with open(dst_path, "w") as f:
        for file in file_list:
            # 判断文件是否以txt结尾，如果不是txt结尾，跳过该文件
            if not file.endswith(".txt"):
                continue
            file_path = os.path.join(output_path, file)
            with open(file_path, "r") as img_txt:
                f.write(os.path.join(img_path, file.replace(".txt", ".jpg")))
                while True:
                    # 读取txt中的内容进行写入
                    line = img_txt.readline()
                    # print(line)
                    f.write(" " + line.strip())
                    if not line:
                        f.write("\n")
                        print("已合并数目：{}".format(index))
                        index += 1
                        break
    print("合并成功，保存路径为：{}".format(dst_path))


if __name__ == '__main__':
    output_path = input("输入需要合并的txt文件夹路径，【默认output/YOLO_darknet】：")
    if output_path == "":
        output_path = "output/YOLO_darknet"
    img_path = input(r"输入训练时的图片路径，【默认D:\PycharmProjects2018.3.5\armlogic_old\keras-yolo3-02\OpenLabeling\main\input】：")
    if img_path == "":
        img_path = r"D:\PycharmProjects2018.3.5\armlogic_old\keras-yolo3-02\OpenLabeling\main\input"
    merge_anno(output_path, img_path)