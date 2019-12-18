
IMG_HEIGHT = 416
IMG_WIDTH = 416

CATEGORY = ['Farmer',#农夫山泉
            'Cola',#可口可乐
            'White sprite',#白色雪碧
            'Vitality',#元气
            'Sprite Green',#绿色雪碧
            'Pulsation',#脉动
            'Centenary mountain',#百岁山
            'Words of the sea',#海之言
            'Chun xiang',#淳享
            'WOW',#哇哈哈
            'Yi Bao'#怡宝
            ]
CLASS_NUM = len(CATEGORY)

"""
32、16、8表示降采样比例
如果输入图像尺寸为416*416，则32代表13*13的尺寸、16代表26*26的尺寸、8代表52*52的尺寸
"""
ANCHORS_GROUP = {
    32: [[116, 90], [156, 198], [373, 326]],
    16: [[30, 61], [62, 45], [59, 119]],
    8: [[10, 13], [16, 30], [33, 23]]
}

ANCHORS_GROUP_AREA = {
    32: [x * y for x, y in ANCHORS_GROUP[13]],
    16: [x * y for x, y in ANCHORS_GROUP[26]],
    8: [x * y for x, y in ANCHORS_GROUP[52]],
}
