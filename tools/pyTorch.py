import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from tools.mnist import *
input_size0 = 28*28 #28*28
hidden_size0 = 500
num_classes = 10

weightsTrainedDir = "E:\\1\\pytorch\\net.pkl"

class Neural_net(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(Neural_net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.relu(out)
        out = self.layer2(out)
        return out

net = Neural_net(input_size0, hidden_size0, num_classes)

def train():
    learning_rate = 1e-1
    num_epoches = 5
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate)
    lossF = 1000
    for epoch in range(num_epoches):
        print("current epoch = %d" % epoch)
        for i, (images, labels) in enumerate(train_loader):#train_loader 是一个批次的数据
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
            outputs = net(images)
            # print("outputs", outputs)
            # print("outputsshape::", outputs.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward() #
            optimizer.step() #更新权重
            lossF = loss.item()
            if i%100 == 0:
                print("current loss = %.5f" % loss.item())
    print("finished training,loss: ", lossF)


def test():
    total = 0
    correct = 0
    for images, labels in test_loader:# 一批次一批次的看
        images = Variable(images.view(-1, 28*28))
        outputs = net(images)
        _,predicts = torch.max(outputs.data, 1)
        # print("predict", predicts)
        # print(labels.size(0))
        total += labels.size(0)
        correct += (predicts == labels).sum()
    # print("accuracy = %.2f" % (100*correct/total))

def findTheNumPic(mytest0, left, top, w, h):
    #初始化参数
    kernel3 = np.ones((3, 3), np.uint8)
    show= mytest0.copy()
    #一步剪切出来大致位置
    mytest = cv2.cvtColor(mytest0[top: top + h, left: left + w], cv2.COLOR_BGR2GRAY)
    # cv2.imshow("mytest", mytest)
    cv2.rectangle(show, (left, top), (left+w, top+h), (0, 0, 255))
    #二值化然后膨胀然后边缘然后检测轮廓
    ret, threshed = cv2.threshold(mytest, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("threshed", threshed)
    dilated = cv2.dilate(threshed, kernel3)

    # cv2.imshow("dilated", dilated)

    edge = cv2.Canny(dilated, 78, 148)

    # cv2.imshow("edge", edge)

    if cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    elif cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(show, contours, -1, (0, 255, 0), 1)
#找出轮廓面积最大的 就是数字图片
    arealist = []
    if len(contours) > 0:
        for ci in range(len(contours)):
            arclenth = cv2.arcLength(contours[ci], True)  # 面积
            area = cv2.contourArea(contours[ci])  # 4386.5
            arealist.append(area)
    #         print("area = ", area)
    # print(arealist)
    sortIndex = sorted(range(len(arealist)), key=lambda k: arealist[k], reverse=True)
    # print(sortIndex)

    # cv2.drawContours(show, contours, sortIndex[0], (0, 255, 255), 1)
    #找出最大的轮廓的外接矩形 并抠出来
    contourBndBox = cv2.boundingRect(contours[sortIndex[0]])  # x,y,w,h  外接矩形

    x = contourBndBox[0]+left
    y = contourBndBox[1]+top
    w = contourBndBox[2]
    h = contourBndBox[3]
    cv2.rectangle(show, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 画矩形

    res = mytest0[y:y+h, x:x+w].copy()
    res = cv2.copyMakeBorder(res, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])#扩充边界
    # cv2.imshow("show", show)
    # cv2.imshow("res", res)

    ret, num_pic = cv2.threshold(res, 100, 255, cv2.THRESH_BINARY)
    mytest = 255 - cv2.cvtColor(num_pic, cv2.COLOR_BGR2GRAY)

    mytest = cv2.resize(mytest, (28, 28), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow("my", mytest)
    return mytest, x, y, w, h
    # cv2.imshow("edge", edge)


def numRecogByMnistKnn(num_pic):
    ret, num_pic = cv2.threshold(num_pic, 100, 255, cv2.THRESH_BINARY)
    mytest = 255 - cv2.cvtColor(num_pic, cv2.COLOR_BGR2GRAY)

    mytest = cv2.resize(mytest, (28, 28), interpolation=cv2.INTER_CUBIC)

    cv2.imshow("my", mytest)
    return mytest

def torchPred(pic):
    net = torch.load(weightsTrainedDir)
    torch_data = torch.from_numpy(pic)
    torch_data=torch_data.float() #防止报错
    # print(torch_data.shape)
    torch_data  = Variable(torch_data.view(-1, 28 * 28))
    # print(torch_data.shape)
    outputs = net(torch_data)
    _, predicts = torch.max(outputs.data, 1)
    # print("predict", predicts.numpy()[0])
    return  predicts.numpy()[0]



if __name__ == '__main__':
    # train()
    # torch.save(net, "E:\\1\\pytorch\\net.pkl")
    net = torch.load("E:\\1\\pytorch\\net.pkl")

    my = cv2.imread("E:\\1\\1\\6.jpg")

    # left = 530
    # top = 280
    # w = 70
    # h = 90
    # numPic, x0, y0, w0, h0 = findTheNumPic(my, left, top, w, h)
    #
    # cv2.rectangle(my, (x0, y0), (x0 + w0, y0 + h0), (255, 255, 0), 2)  # 画矩形
    # cv2.imshow("jpg", numPic)


    # test()
    pic = numRecogByMnistKnn(my)

    torch_data = torch.from_numpy(pic)

    torch_data=torch_data.float() #防止报错
    print(torch_data.shape)
    torch_data  = Variable(torch_data.view(-1, 28 * 28))
    print(torch_data.shape)
    outputs = net(torch_data)
    _, predicts = torch.max(outputs.data, 1)
    print("predict", predicts.numpy()[0])

    # total = 0
    # correct = 0
    # for images, labels in test_loader:  # 一批次一批次的看
    #     images = Variable(images.view(-1, 28 * 28))
    #     outputs = net(images)
    #     _, predicts = torch.max(outputs.data, 1)
    #     # print("predict", predicts)
    #     print(labels.size(0))
    #     total += labels.size(0)
    #     correct += (predicts == labels).sum()
    #


    cv2.waitKey(0)








