import glob
from pathlib import Path
import matplotlib.pyplot as plt
import tqdm

from sklearn import preprocessing

import cv2 as cv
import numpy as np

class VisualOdometry:
    def __init__(self, skip=None, corners=None, quality=None, min=None):
        if skip is None:
            self.skipFrames = 2#7#5#3#5 #3
            self.maxCorners = 1600
            self.qualitlyLevel = 0.05
            self.minDistance = 10
        else:
            self.skipFrames = skip
            self.maxCorners = corners
            self.qualitlyLevel = quality
            self.minDistance = min

        self.intrinsicParameters = []
        with open('./VO Practice Sequence/VO Practice Camera Parameters.txt') as f:
            # reading each line
            for line in f:
                thisRow = []
                # reading each word
                for word in line.split():
                    thisRow.append(float(word))

                thisRow = np.array(thisRow)
                self.intrinsicParameters.append(thisRow)

        self.intrinsicParameters = np.array(self.intrinsicParameters)
        self.distortionCoefficent = None
        self.prevFrame = None


        # template matching parameters
        self.templateWidth = 70
        self.templateHeight = self.templateWidth
        self.windowHeightWidth = 200

    def makeTemplates(self, points, image):
        templates = []
        # x_orig_max = image.shape[0]
        for point in points:
            if point is None:
                continue
            xValue = int(point[0])
            yValue = int(point[1])
            ymin = yValue - int((self.templateHeight)/2)
            ymax = yValue + int((self.templateHeight)/2)
            xmin = xValue - int(self.templateWidth / 2)
            xmax = xValue + int(self.templateWidth / 2)

            if ymin < 0:
                ymin = 0
                ymax = self.templateHeight - 1
            elif ymax >= image.shape[0]:
                ymax = image.shape[0] - 1
                ymin = ymax - self.templateHeight

            if xmin < 0:
                xmin = 0
                xmax = self.templateWidth - 1
            elif xmax >= image.shape[1]:
                xmax = image.shape[1] - 1
                xmin = xmax - self.templateWidth

            roi = image[ymin:ymax, xmin:xmax]

            templates.append(roi)

        return np.array(templates)

    # def templateMatching(self, pathToVid):
    #     picList = list(glob.glob(pathToVid + "/**.png"))
    #     picList.sort()
    #     print(picList)
    #
    #     frame = cv.imread(picList[0])
    #
    #     saveAt = Path("./Results")
    #     saveAt.mkdir(exist_ok=True)
    #     titleName = "task2_"
    #     cntr = 0
    #     saveAs = ".avi"
    #     saveHere = saveAt / (titleName + str(cntr) + saveAs)
    #     while saveHere.exists():
    #         cntr += 1
    #         saveHere = saveAt / (titleName + str(cntr) + saveAs)
    #
    #     w = int(frame.shape[0])
    #     h = int(frame.shape[1])
    #
    #     # video recorder
    #     fourcc = cv.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
    #
    #     myRecorder = cv.VideoWriter(str(saveHere), fourcc, 30, (w, h))
    #
    #     prevGray_list = []
    #     prevPoints_list = []
    #     templates_list = []
    #     for x in range(self.skipFrames):
    #         prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #         prevPoints = cv.goodFeaturesToTrack(prevGray, maxCorners=400, qualityLevel=0.25, minDistance=20)
    #         prevPoints = np.array(prevPoints)
    #         prevPoints = np.squeeze(prevPoints, axis=1)
    #         templates = self.makeTemplates(prevPoints, frame)
    #         prevGray_list.append(prevGray)
    #         prevPoints_list.append(prevPoints)
    #         templates_list.append(templates)
    #
    #     overall_cntr = 0
    #     cntr = 0
    #     for pic in picList:
    #         frame = cv.imread(pic)
    #
    #         frame32 = frame.astype(np.float32)
    #
    #         newPoints = []
    #
    #         self.cameraPoseMotion(str(pathToVid), overall_cntr, overall_cntr + 1)
    #         overall_cntr += 1
    #
    #         for i in range(len(templates_list[cntr])):
    #             template = templates_list[0][i]
    #             oldPoint = prevPoints_list[cntr][i]
    #
    #             template = template.astype(np.float32)
    #             newPoint = self.windowedTemplate(oldPoint, template, frame32.copy())
    #             newPoints.append(newPoint)
    #
    #         newPoints = np.array(newPoints)
    #
    #         # draw the tracks
    #         for i, (new, old) in enumerate(zip(newPoints, prevPoints_list[cntr])):
    #             if new is None:
    #                 continue
    #             a, b = new[0], new[1]
    #             c, d = old[0], old[1]  #
    #             frame = cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
    #             frame = cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
    #         cv.imshow("Current Frame", frame)
    #
    #         if cv.waitKey(1) & 0xFF == ord('a'):
    #             break
    #         myRecorder.write(frame)
    #
    #         prevPoints_list[cntr] = newPoints
    #         # templates_list[cntr] = self.makeTemplates(newPoints, frame)
    #         cntr += 1
    #         if cntr >= self.skipFrames:
    #             cntr = 0
    #
    #     # at the end release
    #     myRecorder.release()

    def windowedTemplate(self, oldPoint, template, frame):
        # x_max_orig = frame.shape[0]
        if oldPoint is None:
            return None
        xValue = int(oldPoint[0])
        yValue = int(oldPoint[1])
        ymin = yValue - int((self.windowHeightWidth) / 2)
        ymax = yValue + int((self.windowHeightWidth) / 2)
        xmin = xValue - int(self.windowHeightWidth / 2)
        xmax = xValue + int(self.windowHeightWidth / 2)

        if ymin < 0:
            ymin = 0
            ymax = self.windowHeightWidth - 1
        elif ymax >= frame.shape[0]:
            ymax = frame.shape[0] - 1
            ymin = ymax - self.windowHeightWidth

        if xmin < 0:
            xmin = 0
            xmax = self.windowHeightWidth - 1
        elif xmax >= frame.shape[1]:
            xmax = frame.shape[1] - 1
            xmin = xmax - self.windowHeightWidth

        frame = frame[ymin:ymax, xmin:xmax]
        result = cv.matchTemplate(frame, template, cv.TM_SQDIFF_NORMED)

        cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)


        result = (min_loc[0] + xmin + int((self.templateHeight)/2), min_loc[1] + ymin + int((self.templateHeight)/2))

        if abs(min_val) > 0.9 * 10 ** (-7):
            return None
        return result

    def cameraPoseMotion(self, vidPath, firstFrame, secondFrame, myVid = None, totalVid = None):
        # print("enters")
        self.firstFrame_index = firstFrame#100#240
        self.secondFrame_index = secondFrame#150#275


        # myVid = cv.VideoCapture(vidPath)
        #
        # ret, frame = myVid.read()

        if myVid is not None:
            # myVid = cv.VideoCapture(vidPath)
            if self.prevFrame is None:
                ret, frame = myVid.read()
                prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # print("prevGray here", prevGray)
            else:

                frame = self.prevFrame
                prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # print("here", prevGray)
        else:
            thePicPath = (glob.glob(vidPath + "/**00" + str(firstFrame) + ".png"))[0]
            frame = cv.imread(thePicPath)
            prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # cv.imwrite("./Results/task3_frame1.jpg", frame)

        # for x in range(self.skipFrames):

        prevPoints = cv.goodFeaturesToTrack(prevGray, maxCorners=self.maxCorners, qualityLevel=self.qualitlyLevel, minDistance=self.minDistance)

        if myVid is not None:
            # myVid = cv.VideoCapture(vidPath)

            ret, frame = myVid.read()
            # print(frame.shape)
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if not ret:
                frame = self.prevFrame
                nextGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                return None, None
            else:
                self.prevFrame = frame.copy()
                nextGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            thePicPath = (glob.glob(vidPath + "/**00" + str(secondFrame) + ".png"))
            if len(thePicPath):
                frame = cv.imread(thePicPath[0])
            else:
                print("got last frame")
                frame = cv.imread(glob.glob(vidPath + "/**00" + str(totalVid) + ".png")[0])
        # print(len(thePicPath))
        # frame = cv.imread(thePicPath)

            nextGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # print(prevGray)
        # print("between")
        # print(nextGray)
        # print("before here")

        nextPoints, status, error = cv.calcOpticalFlowPyrLK(prevGray, nextGray,
                                                             prevPts=prevPoints, nextPts=None)
        # print("after here")
        # cv.imwrite("./Results/task3_frame2.jpg", frame)

        # Select good points
        if nextPoints is not None:
            good_new = nextPoints[status == 1]
            good_old = prevPoints[status == 1]
        # undistort the points
        new_undistorted = []
        old_undistorted = []

        thisSum = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            thisSum.append(abs(sum(new-old)))

        thisSum.sort()
        savePercentile = 0.50
        cutIndex = int(len(thisSum) * savePercentile)
        dynamicThresh = thisSum[cutIndex]
        # print(dynamicThresh)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # new = cv.undistortPoints(new, self.intrinsicParameters, self.distortionCoefficent)
            # old = cv.undistortPoints(old, self.intrinsicParameters, self.distortionCoefficent)
            # print(abs(sum(new - old)))
            if abs(sum(new-old)) <= dynamicThresh: #pull out noisy ones
                # print("adding", abs(sum(new - old)))
                new_undistorted.append(new)
                old_undistorted.append(old)


        new_undistorted = np.array(new_undistorted)
        old_undistorted = np.array(old_undistorted)

        # new_undistorted = np.squeeze(new_undistorted, axis=2)
        # old_undistorted = np.squeeze(old_undistorted, axis=2)

        f, mask = cv.findFundamentalMat(new_undistorted, old_undistorted, cv.FM_8POINT)
        # print("F Matrix: ", f)
        # print("the f", f)
        # print("new", new_undistorted)
        # print("old", old_undistorted)
        # print("the mask", mask)
        if f is None:
            return None, None
        E = np.dot(np.dot(np.transpose(self.intrinsicParameters), f), self.intrinsicParameters)
        E = preprocessing.normalize(E)
        # print("Essential Matrix, E:", E)

        pts, R, t, mask = cv.recoverPose(E, new_undistorted, old_undistorted)
        self.displayHelp(new_undistorted, old_undistorted, frame)

        # print("Rotation Matrix, R:", R)
        # print("Rotation Matrix, R (3x1):", cv.Rodrigues(R)[0])
        # print("Translation Matrix, T:", t)

        # myVid.release()

        self.Rotation = R
        self.Translation = t

        return t, R

    def displayHelp(self, newPoints, prevPoints, frame):
        # draw the tracks
        for i, (new, old) in enumerate(zip(newPoints, prevPoints)):
            if new is None:
                continue
            a, b = new[0], new[1]
            c, d = old[0], old[1]  #
            frame = cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
        cv.imshow("Current Frame", frame)
        cv.waitKey(1)

    def drawGraph(self, allPoints):
        x = []
        # corresponding y axis values
        y = []

        # test =
        prevX = 0
        prevZ = 0
        for point in allPoints:
            # print(thisX)
            x_t = point[0][0]
            y_t = point[1][0]
            z_t = point[2][0]

            thisX = (x_t / y_t) + prevX
            thisZ = (z_t / y_t) + prevZ
            x.append(thisX)
            y.append(thisZ)

            prevX = thisX
            prevZ = thisZ


        # plotting the points
        plt.plot(x, y)

        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')

        # giving a title to my graph
        plt.title('The Graph')

        saveHere = "./Results/plot_" + str(self.skipFrames) + "_" + str(self.maxCorners) + "_" + str(self.qualitlyLevel) + "_" + str(self.minDistance) + "_"
        saveAs = ".png"
        cntr = 0

        while Path(saveHere + str(cntr) + saveAs).exists():
            cntr += 1
        plt.savefig(saveHere + str(cntr) + saveAs)

        plt.figure()

    def calculateAll(self, pathToVid, totalVid):
        saveAt = Path("./Results")
        saveAt.mkdir(exist_ok=True)
        titleName = "R_T_Calculated_" + str(self.skipFrames) + "_" + str(self.maxCorners) + "_" + str(self.qualitlyLevel) + "_" + str(self.minDistance) + "_"
        cntr = 0
        saveAs = ".txt"
        saveHere = saveAt / (titleName + str(cntr) + saveAs)
        while saveHere.exists():
            cntr += 1
            saveHere = saveAt / (titleName + str(cntr) + saveAs)

        if ".avi" in pathToVid:
            myVid = cv.VideoCapture(pathToVid)
        else:
            myVid = None

        allTranslation = []
        writeList = []

        for overall_cntr in (range(0, totalVid)):
            # print("start")
            # print(str(pathToVid))
            # print(overall_cntr)
            # print(self.skipFrames)
            # print(myVid)
            translation, rotation = self.cameraPoseMotion(str(pathToVid), overall_cntr, overall_cntr + self.skipFrames, myVid, totalVid)
            if translation is None:
                print("end at ", overall_cntr)
                break
            saveThis = str(rotation[0][0]) + " " + str(rotation[0][1]) + " " + str(rotation[0][2]) + " "
            saveThis += str(translation[0][0]) + " "
            saveThis += str(rotation[1][0]) + " " + str(rotation[1][1]) + " " + str(rotation[1][2]) + " "
            saveThis += str(translation[1][0]) + " "
            saveThis += str(rotation[2][0]) + " " + str(rotation[2][1]) + " " + str(rotation[2][2]) + " "
            saveThis += str(translation[2][0]) +"\n"

            writeList.append(saveThis)

            # print("saveThis", saveThis)
            # print("rotation", rotation)
            #
            # print("translation", overall_cntr, translation)
            allTranslation.append(translation)

        with open(str(saveHere), 'w') as f:
            f.writelines(writeList)

        self.drawGraph(allTranslation)


