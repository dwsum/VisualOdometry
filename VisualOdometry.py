import glob
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn import preprocessing

import cv2 as cv
import numpy as np

class VisualOdometry:
    def __init__(self):
        self.skipFrames = 2#7#5#3#5 #3
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

        print(self.intrinsicParameters)
        self.intrinsicParameters = np.array(self.intrinsicParameters)
        print("start", self.intrinsicParameters)
        print("start2", type(self.intrinsicParameters))
        self.distortionCoefficent = None


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

    def templateMatching(self, pathToVid):
        picList = list(glob.glob(pathToVid + "/**.png"))
        picList.sort()
        print(picList)

        frame = cv.imread(picList[0])

        saveAt = Path("./Results")
        saveAt.mkdir(exist_ok=True)
        titleName = "task2_"
        cntr = 0
        saveAs = ".avi"
        saveHere = saveAt / (titleName + str(cntr) + saveAs)
        while saveHere.exists():
            cntr += 1
            saveHere = saveAt / (titleName + str(cntr) + saveAs)

        w = int(frame.shape[0])
        h = int(frame.shape[1])

        # video recorder
        fourcc = cv.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist

        myRecorder = cv.VideoWriter(str(saveHere), fourcc, 30, (w, h))


        prevGray_list = []
        prevPoints_list = []
        templates_list = []
        for x in range(self.skipFrames):
            prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            prevPoints = cv.goodFeaturesToTrack(prevGray, maxCorners=400, qualityLevel=0.25, minDistance=20)
            prevPoints = np.array(prevPoints)
            prevPoints = np.squeeze(prevPoints, axis=1)
            templates = self.makeTemplates(prevPoints, frame)
            prevGray_list.append(prevGray)
            prevPoints_list.append(prevPoints)
            templates_list.append(templates)


        overall_cntr = 0
        cntr = 0
        for pic in picList:
            frame = cv.imread(pic)

            frame32 = frame.astype(np.float32)

            newPoints = []

            self.cameraPoseMotion(str(pathToVid), overall_cntr, overall_cntr + 1)
            overall_cntr += 1

            for i in range(len(templates_list[cntr])):
                template = templates_list[0][i]
                oldPoint = prevPoints_list[cntr][i]

                template = template.astype(np.float32)
                newPoint = self.windowedTemplate(oldPoint, template, frame32.copy())
                newPoints.append(newPoint)

            newPoints = np.array(newPoints)

            # draw the tracks
            for i, (new, old) in enumerate(zip(newPoints, prevPoints_list[cntr])):
                if new is None:
                    continue
                a, b = new[0], new[1]
                c, d = old[0], old[1]#
                frame = cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
            cv.imshow("Current Frame", frame)

            if cv.waitKey(1) & 0xFF == ord('a'):
                break
            myRecorder.write(frame)

            prevPoints_list[cntr] = newPoints
            # templates_list[cntr] = self.makeTemplates(newPoints, frame)
            cntr += 1
            if cntr >= self.skipFrames:
                cntr = 0

        #at the end release
        myRecorder.release()


    def cameraPoseMotion(self, vidPath, firstFrame, secondFrame):
        self.firstFrame_index = firstFrame#100#240
        self.secondFrame_index = secondFrame#150#275


        # myVid = cv.VideoCapture(vidPath)
        #
        # ret, frame = myVid.read()

        thePicPath = (glob.glob(vidPath + "/**00" + str(firstFrame) + ".png"))[0]
        frame = cv.imread(thePicPath)

        cv.imwrite("./Results/task3_frame1.jpg", frame)

        # for x in range(self.skipFrames):
        prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        prevPoints = cv.goodFeaturesToTrack(prevGray, maxCorners=800, qualityLevel=0.04, minDistance=10)

        thePicPath = (glob.glob(vidPath + "/**00" + str(secondFrame) + ".png"))[0]
        frame = cv.imread(thePicPath)

        nextGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        nextPoints, status, error = cv.calcOpticalFlowPyrLK(prevGray, nextGray,
                                                             prevPts=prevPoints, nextPts=None)

        cv.imwrite("./Results/task3_frame2.jpg", frame)

        # Select good points
        if nextPoints is not None:
            good_new = nextPoints[status == 1]
            good_old = prevPoints[status == 1]
        # undistort the points
        new_undistorted = []
        old_undistorted = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # new = cv.undistortPoints(new, self.intrinsicParameters, self.distortionCoefficent)
            # old = cv.undistortPoints(old, self.intrinsicParameters, self.distortionCoefficent)
            new_undistorted.append(new)
            old_undistorted.append(old)


        new_undistorted = np.array(new_undistorted)
        old_undistorted = np.array(old_undistorted)

        # new_undistorted = np.squeeze(new_undistorted, axis=2)
        # old_undistorted = np.squeeze(old_undistorted, axis=2)

        f, mask = cv.findFundamentalMat(new_undistorted, old_undistorted, cv.FM_8POINT)
        # print("F Matrix: ", f)

        E = np.dot(np.dot(np.transpose(self.intrinsicParameters), f), self.intrinsicParameters)
        E = preprocessing.normalize(E)
        # print("Essential Matrix, E:", E)

        pts, R, t, mask = cv.recoverPose(E, new_undistorted, old_undistorted)

        # print("Rotation Matrix, R:", R)
        # print("Rotation Matrix, R (3x1):", cv.Rodrigues(R)[0])
        # print("Translation Matrix, T:", t)

        # myVid.release()

        self.Rotation = R
        self.Translation = t

        return t, cv.Rodrigues(R)[0]

    def drawGraph(self, allPoints):
        x = []
        # corresponding y axis values
        y = []

        # test =
        prevX = 0
        prevZ = 0
        for point in allPoints:
            # print(thisX)

            thisX = point[0][0] + prevX
            thisZ = abs(point[2][0]) + prevZ
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

        saveHere = "./Results/plot_"
        saveAs = ".png"
        cntr = 0

        while Path(saveHere + str(cntr) + saveAs).exists():
            cntr += 1
        plt.savefig(saveHere + str(cntr) + saveAs)

    def calculateAll(self, pathToVid):
        picList = list(glob.glob(pathToVid + "/**.png"))
        picList.sort()
        print(picList)

        frame = cv.imread(picList[0])

        saveAt = Path("./Results")
        saveAt.mkdir(exist_ok=True)
        titleName = "task2_"
        cntr = 0
        saveAs = ".avi"
        saveHere = saveAt / (titleName + str(cntr) + saveAs)
        while saveHere.exists():
            cntr += 1
            saveHere = saveAt / (titleName + str(cntr) + saveAs)

        w = int(frame.shape[0])
        h = int(frame.shape[1])

        # video recorder
        fourcc = cv.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist

        # myRecorder = cv.VideoWriter(str(saveHere), fourcc, 30, (w, h))


        # prevGray_list = []
        # prevPoints_list = []
        # templates_list = []
        # for x in range(self.skipFrames):
        #     prevGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #     prevPoints = cv.goodFeaturesToTrack(prevGray, maxCorners=400, qualityLevel=0.25, minDistance=20)
        #     prevPoints = np.array(prevPoints)
        #     prevPoints = np.squeeze(prevPoints, axis=1)
        #     templates = self.makeTemplates(prevPoints, frame)
        #     prevGray_list.append(prevGray)
        #     prevPoints_list.append(prevPoints)
        #     templates_list.append(templates)

        allTranslation = []
        # overall_cntr = 0
        cntr = 0

        for overall_cntr in range(0, 700):
            translation, rotation = self.cameraPoseMotion(str(pathToVid), overall_cntr, overall_cntr + self.skipFrames)

            print("translation", overall_cntr, translation)
            allTranslation.append(translation)

        # print(allTranslation)
        self.drawGraph(allTranslation)
        # for pic in picList:
        #     frame = cv.imread(pic)
        #
        #     frame32 = frame.astype(np.float32)
        #
        #     newPoints = []
        #
        #     translation, rotation = self.cameraPoseMotion(str(pathToVid), overall_cntr, overall_cntr + 1)
        #
        #     print("translation", overall_cntr, translation)
        #     allTranslation.append(translation)
        #     overall_cntr += 1
        #
        #     for i in range(len(templates_list[cntr])):
        #         template = templates_list[0][i]
        #         oldPoint = prevPoints_list[cntr][i]
        #
        #         template = template.astype(np.float32)
        #         newPoint = self.windowedTemplate(oldPoint, template, frame32.copy())
        #         newPoints.append(newPoint)
        #
        #     newPoints = np.array(newPoints)
        #
        #     # draw the tracks
        #     for i, (new, old) in enumerate(zip(newPoints, prevPoints_list[cntr])):
        #         if new is None:
        #             continue
        #         a, b = new[0], new[1]
        #         c, d = old[0], old[1]#
        #         frame = cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
        #         frame = cv.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
        #     cv.imshow("Current Frame", frame)
        #
        #     if cv.waitKey(1) & 0xFF == ord('a'):
        #         break
        #     myRecorder.write(frame)
        #
        #     prevPoints_list[cntr] = newPoints
        #     # templates_list[cntr] = self.makeTemplates(newPoints, frame)
        #     cntr += 1
        #     if cntr >= self.skipFrames:
        #         cntr = 0
        #
        # #at the end release
        # myRecorder.release()



