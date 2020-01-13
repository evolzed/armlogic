import os

import json
# print(cuPath)
# print(fp)
un_anno = []


def json2anno(json_path="./output/jsons", anno_out_path="output/bottle_annotation.txt"):
    """
    自动生成训练需要的文件格式

    :param json_path: json文件的存放位置，若不传入，默认是当前目录下的./output/jsons
    :param anno_out_path: 生成annotation文件的存放位置，默认为./output/bottle_annotation.txt
    :return: None
    """
    fp = os.listdir(json_path)
    cuPath = os.path.abspath(json_path)
    n = 0
    with open(anno_out_path, "w") as anno_file:
        for i in fp:
            if i.endswith(".json"):
                # print(os.path.join(cuPath, i))
                with open(cuPath + "/" + i, "r") as f:
                    # con = f.readlines()
                    dic = json.loads(f.read())
                    # windows 下解决中文问题
                    # dic = json.loads(f.read().encode("gbk").decode("utf8"))
                    # print(type(dic))
                    outputs = dic.get("outputs")
                    if outputs is None:
                        print("--None--" * 20, i)
                        continue
                    try:
                        obj = outputs["object"]
                    except Exception as e:
                        print(e, i)
                        un_anno.append(i)
                        continue
                    context = ""
                    for bbox in obj:
                        bbox_class = bbox["name"]
                        bbox_xmin = str(bbox["bndbox"]["xmin"] if bbox["bndbox"]["xmin"] >= 0 else 0)
                        bbox_ymin = str(bbox["bndbox"]["ymin"] if bbox["bndbox"]["ymin"] >= 0 else 0)
                        bbox_xmax = str(bbox["bndbox"]["xmax"] if bbox["bndbox"]["xmax"] <= 1280 else 1280)
                        bbox_ymax = str(bbox["bndbox"]["ymax"] if bbox["bndbox"]["ymax"] <= 960 else 960)
                        context += bbox_xmin + "," + bbox_ymin + "," + bbox_xmax + "," + bbox_ymax + "," + bbox_class + " "
                    anno_file.write(os.path.join(cuPath, (i.replace(".json", ".jpg ")) + context + "\n"))
                    n += 1
            print(i)
        print("共修改{}个文件！".format(n))
        print("可能未标注的照片{}个,分别是{}".format((len(fp)-n), un_anno))

        # with open(cuPath + "/" + i, "w") as f:
        #     for c in con:
        #         con = c.strip().replace(" ", ",")
        #         f.write(con + "\n")
        # print("改写{}个文件成功！".format(n+1))
        # n += 1


if __name__ == '__main__':
    json_p = "./output/jsons"
    anno_p = "./output/bottle_annotation.txt"
