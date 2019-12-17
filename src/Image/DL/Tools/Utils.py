import numpy as np

'''转换独热编码'''


def OneHot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


'''交并比'''


def IOU(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    '''计算面积'''

    left_x = np.maximum(box[0], boxes[:, 0])
    up_y = np.maximum(box[1], boxes[:, 1])
    right_x = np.minimum(box[2], boxes[:, 2])
    low_y = np.minimum(box[3], boxes[:, 3])
    '''找交集'''

    w = np.maximum(0, right_x - left_x)
    h = np.maximum(0, low_y - up_y)
    '''判断交集情况'''

    inter = w * h
    if isMin:
        return np.true_divide(inter, np.minimum(box_area, boxes_area))
    else:
        return np.true_divide(inter, (box_area + boxes_area - inter))
    return np.maximum(min, normal)
    '''返回结果'''


'''非极大值抑制'''


def NMS(boxes, thresh=0.3, isMin=False, idx=1):
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
