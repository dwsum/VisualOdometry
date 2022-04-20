import glob
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn import preprocessing

import cv2 as cv
import numpy as np

class VisualOdometry:
    def __init__(self, skip=None, corners=None, quality=None, min=None):
        point_count = 10000
        self.sift = cv.xfeatures2d.SIFT_create(point_count)
        self.bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)


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

        for point in allPoints:

            x_t = point[0]
            y_t = point[1]
            z_t = point[2]

            thisX = x_t
            thisZ = z_t

            if not np.isnan(thisX):
                x.append(thisX)
            if not np.isnan(thisZ):
                y.append(thisZ)

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

        plt.axis("equal")
        plt.savefig(saveHere + str(cntr) + saveAs)

        plt.figure()

    def refinePoints(self, matches):
        good_points = []
        for m, n in matches:
            # append the points according
            # to distance of descriptors

            if (m.distance < 0.5 * n.distance):
                good_points.append(m)
        return good_points

    def calculateAll_sift(self, pathToVid, totalVid):
        saveAt = Path("./Results")
        saveAt.mkdir(exist_ok=True)
        titleName = "R_T_Calculated_"
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

        pos = np.array([[0, 0, 0, 1]]).T
        pos = [pos]
        allTranslation = [pos]

        position_list = [np.zeros(3)]
        rotation_list = [np.eye(3)]

        prev_kps = None
        prev_desc = None

        pathToVid = Path(pathToVid)
        for vid in tqdm(sorted(pathToVid.iterdir())):
            # print(vid)
            frame = cv.imread(str(vid))

            # sift
            keypoints_1, descriptors_1 = self.sift.detectAndCompute(frame, None)

            if prev_kps is not None and prev_desc is not None:
                matches = self.flann.knnMatch(prev_desc, descriptors_1, k=2)

                points = self.refinePoints(matches)
                # print("points", points)
                prev_pts = np.float32([prev_kps[m.queryIdx]
                                      .pt for m in points]).reshape(-1, 1, 2)

                if len(prev_pts) > 1:
                    new_pts = np.float32([keypoints_1[m.trainIdx].pt for m in points]).reshape(-1, 1, 2)


                    f, mask = cv.findFundamentalMat(new_pts, prev_pts, cv.RANSAC, 5.0, (1 - (1e-5)), cv.FM_RANSAC)

                    E = np.dot(np.dot(np.transpose(self.intrinsicParameters), f), self.intrinsicParameters)
                    # E = preprocessing.normalize(E)  # TODO maybe this line wrong
                    # print("Essential Matrix, E:", E)

                    u, v, vt = np.linalg.svd(E)
                    sigma = np.diag(np.array([1, 1, 0]))
                    E_norm = u @ sigma @ vt
                    pts, rotation, translation, mask = cv.recoverPose(E_norm, new_pts, prev_pts, self.intrinsicParameters, mask=mask)

                    transform = np.block([[rotation, translation], [0, 0, 0, 1]])
                    transform = np.linalg.inv(transform)
                    allTranslation.append(transform @ allTranslation[-1])

                    rotation_list.append(rotation @ rotation_list[-1])
                    position_list.append(
                        position_list[-1] + (rotation_list[-1]) @ translation.reshape((3, 1)).reshape(-1))


            prev_kps = keypoints_1
            prev_desc = descriptors_1





        #TODO uncomment this to write txt file
        # with open(str(saveHere), 'w') as f:
        #     f.writelines(writeList)

        self.drawGraph(position_list)



