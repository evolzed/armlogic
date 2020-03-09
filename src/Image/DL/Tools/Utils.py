import numpy as np


def OneHot(cls_num, v):
    """转换独热编码"""
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


def IOU(box, boxes, isMin=False):
    """交并比"""
    # 计算面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 找交集
    left_x = np.maximum(box[0], boxes[:, 0])
    up_y = np.maximum(box[1], boxes[:, 1])
    right_x = np.minimum(box[2], boxes[:, 2])
    low_y = np.minimum(box[3], boxes[:, 3])

    # 判断交集情况
    w = np.maximum(0, right_x - left_x)
    h = np.maximum(0, low_y - up_y)

    inter = w * h
    # 返回结果
    if isMin:
        return np.true_divide(inter, np.minimum(box_area, boxes_area))
    else:
        return np.true_divide(inter, (box_area + boxes_area - inter))


def NMS(boxes, thresh=0.3, isMin=False, idx=1):
    """非极大值抑制"""
    args = -boxes[:, 0].argsort()  # 倒序排列
    sort_boxes = boxes[args]  # 获取排序后的数据
    keep_boxes = []  # 存放保留的坐标

    while len(sort_boxes) > 0:
        _box = sort_boxes[0]
        keep_boxes.append(_box)

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]
            _iou = IOU(_box[idx:], _boxes[:, idx:], isMin)  # 计算IOU
            sort_boxes = _boxes[_iou < thresh]  # 保留的数据
        else:
            break

    return np.stack(keep_boxes)
