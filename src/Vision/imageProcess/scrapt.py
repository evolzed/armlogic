
"""
#find the small distance
pointLen = good_new.shape[0]
disarray = np.array([])
for i in range(pointLen):
    dis = self.imgproc.eDistance(good_new[i], good_old[i])
    disarray = np.append(disarray, dis)

# get the low 20% distance point,that is more precision points
reduce = np.percentile(disarray, 40, axis=0)
reducearr = disarray[disarray <= reduce]
index = np.where(disarray <= reduce)
index = index[0]
print(np.array([good_new[0]]))
good_new0 = np.array([[0, 0]])
good_old0 = np.array([[0, 0]])
for i in index:
    good_new0 = np.append(good_new0, np.array([good_new[i]]), axis=0)
    good_old0 = np.append(good_old0, np.array([good_old[i]]), axis=0)
good_new0 = np.delete(good_new0, 0, axis=0)
good_old0 = np.delete(good_old0, 0, axis=0)

good_new = good_new0.copy()
good_old = good_old0.copy()

"""

"""
        if flag == 1:
            good_new, good_old, offset, img = self.imgproc.lkLightflow_track(preframeb, frame, None, None)
        preframe = _frame
        preframeb = frame.copy()
        if "box" in dataDict and dataDict["box"][0][1] > 0.90 and good_new is not None and good_old is not None\
                and dataDict["box"][0][3] > 206 and flag == 1:
            le = dataDict["box"][0][2]
            t = dataDict["box"][0][3]
            r = dataDict["box"][0][4]
            b = dataDict["box"][0][5]
            #filter the points not in the box
            good_new = good_new[(le < good_new[:, 0]) & (good_new[:, 0] < r) &
                                (b < good_new[:, 1]) & (good_new[:, 1] < t)]

            good_old = good_old[(le < good_old[:, 0]) & (good_old[:, 0] < r) &
                                (b < good_old[:, 1]) & (good_old[:, 1] < t)]

            for i, (new, old) in enumerate(zip(good_new, good_old)):  # fold and enumerate with i
                a, b = new.ravel()  # unfoldq c
                c, d = old.ravel()
                img = cv2.circle(img, (a, b), 4, (100, 100, 155), -1)  #color

            flag = 0
            inputCorner = good_new
        if flag == 0:
#                print("0__inputCorner.shape", inputCorner.shape)
            inputCorner, good_old, offset, img = self.imgproc.lkLightflow_track(preframeb, frame, None, inputCorner)
            print("*"*50)
            # speedarray = good_new - good_old
            # pointLen0 = good_new.shape[0]
            # print("offset_diff_array", speedarray)
            # print("offset_array_sum", np.sum(speedarray, axis=0))
            # speed = np.sum(speedarray, axis=0) / pointLen0
            # if np.isnan(speed[0]):
            #     speed = np.array([0, 0])

            if timer()-startt < 5:
                img = cv2.circle(img, (100, 100), 20, (0, 0, 255), -1)  # color
                left = dataDict["box"][0][2]
                top = dataDict["box"][0][3]
                right = dataDict["box"][0][4]
                bottom = dataDict["box"][0][5]
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)  # red is track box
            else:
                img = cv2.circle(img, (100, 100), 20, (255, 0, 0), -1)  # bule
                # print("*" * 50)
                # print(speed[0])
                # print(type(speed[0]))
                # print("*" * 50)
                # if speed[0] is np.nan:
                #     speed = np.array([0, 0])
                #only adapt  the fast speed
                left += 1*int(offset[0]+0.5)
                top += 1*int(offset[1]+0.5)
                right += 1*int(offset[0]+0.5)
                bottom += 1*int(offset[1]+0.5)
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)  # red is track box
                if timer() - startt > 15:
                    startt =timer()


            cv2.rectangle(img, (le, t), (r, b), (255, 0, 0), 2)

            cv2.putText(img, text=str(int(offset[0])), org=(150, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)
            cv2.putText(img, text=str(int(offset[1])), org=(400, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)


        cv2.imshow("my", img)

        print("offset.shape[0]", offset.shape[0])
        if offset.shape[0] == 2:
            print("offset0", offset[0])
            print("offset1", offset[1])
            print("offset", offset)
        cv2.imshow("orig", _frame)
        if "box" in dataDict:
            left = 0
            top = 0
            right = 0
            bottom = 0
            if dataDict["box"][0][1] > 0.9:
                if k < 3:
                    left = dataDict["box"][0][2] + int(offset[0])
                    top = dataDict["box"][0][3] + int(offset[1])
                    right = dataDict["box"][0][4] + int(offset[0])
                    bottom = dataDict["box"][0][5] + int(offset[1])
                    k = k + 5
                else:
                    left += int(offset[0])
                    top += int(offset[1])
                    right += int(offset[0])
                    bottom += int(offset[1])
                    k=k-1

                myshow = _frame.copy()
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                # cv2.circle(img, (dataDict["box"][0][2], dataDict["box"][0][3]), 4, (255, 0, 255))
                cv2.circle(img, (left, top), 4, (0, 255, 255))
        """

# print(dataDict)
# except Exception as e:offset_diff_array
#     # global gState
#     # gState = 3
#     print(e)
#     break

# if len > 1:
#     good_new = np.delete(good_new, len-1, axis=0)
#     good_old = np.delete(good_old, len-1, axis=0)
# print("dis", dis)
# print("good_new", good_new)
# print("good_old", good_old)

# cv2.circle(drawimg, (100, 100), 15, (255, 0, 0), -1)  # blue  detect
# cornersA = cv2.goodFeaturesToTrack(featureimg, mask=None, **feature_params)
"""
# print("cornersA",cornersA)
if cornersA is not None:
    if np.size(cornersA) > 0:
        cornersB, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, None, **lk_params)
        print("cornersB", cornersB)
        good_new = cornersB[st == 1]
        good_old = cornersA[st == 1]
    if good_new is not None:
        if np.size(good_new) > 0:
            good_old = good_new.copy()
            print("good_old", good_old)
            for i, (new, old) in enumerate(zip(good_new, good_old)):  # fold and enumerate with i
                a, b = new.ravel()  # unfold
                c, d = old.ravel()
                # cv2.circle(drawimg, (a, b), 3, (0, 0, 255), -1)

            if "box" in dataDict and dataDict["box"][0][1] > 0.90 and dataDict["box"][0][3] > 180:
                print("in!!!!!!!!!!!!!!!!!!!!!!!!!in!!!!!!!!!!!!!!!")
                le = dataDict["box"][0][2]
                t = dataDict["box"][0][3]
                r = dataDict["box"][0][4]
                b = dataDict["box"][0][5]
                print("le", le)
                print("t", t)
                print("r", r)
                print("b", b)
                # filter the points not in the box
                good_new = good_new[(le < good_new[:, 0]) & (good_new[:, 0] < r) &
                                    (t < good_new[:, 1]) & (good_new[:, 1] < b)]

                good_old = good_old[(le < good_old[:, 0]) & (good_old[:, 0] < r) &
                                    (t < good_old[:, 1]) & (good_old[:, 1] < b)]
                if np.size(good_old) > 0:
                    for i, (new, old) in enumerate(zip(good_new, good_old)):  # fold and enumerate with i
                        a, b = new.ravel()  # unfoldq c
                        c, d = old.ravel()
                        # img = cv2.circle(img, (a, b), 4, (100, 100, 155), -1)  # color

                    print("change!!!!!!!!!!!!!!!!!!!!!!!!!change!!!!!!!!!!!!!!!")
                    flag = 0
                    cv2.circle(drawimg, (100, 100), 15, (255, 0, 0), -1) #blue  detect
"""

# print("offset", offset[0, 0])
# print("offset", offset[0, 1])
#
# print("left", left)
# print("top", top)
# print("right", right)
# print("bottom", bottom)

# cv2.rectangle(drawimg, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
# cv2.putText(drawimg, text=str(ID), org=(int(left), int(top)),
#             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             fontScale=1, color=(0, 255, 255), thickness=2)

# a = int(a)
# b = int(b)
# c = int(c)
# d = int(d)

# print("good_new0", good_new)
# print("good_old0", good_old)
# good_old = good_new.copy()


# print("pregood_new", good_new)
# print("pregood_old", good_old)


# print("st.shape", st.shape)
# print("good_new.shape", p1.shape)
# print("err", err)
# print("left", left)
# print("top", top)
# print("right", right)
# print("bottom", bottom)
# filter the points not in the box

# p0_sigle = p0[(left <= p0[:, :, 0]) & (p0[:, :, 0] <= right) & (top <= p0[:, :, 1]) & (p0[:, :, 1] <= bottom)]  #会改变shape
#                                 px = p0_sigle.reshape(-1, 1, 2)
#                                 p0_sigle = px.copy()