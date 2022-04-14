import tqdm

from VisualOdometry import VisualOdometry

if __name__ == "__main__":
    # skipFrames = [2,3,4,5]  # 7#5#3#5 #3
    # maxCorners = [50]#[800, 1600]
    # qualitlyLevel = [0.1]#[0.01, 0.03, 0.05, 0.07, 0.09]
    # minDistance = [5, 10, 15, 20, 50]
    #
    # for skip in tqdm.tqdm(skipFrames):
    #     for corner in maxCorners:
    #         for quality in qualitlyLevel:
    #             for minDist in minDistance:
    #                 myVO = VisualOdometry(skip, corner, quality, minDist)
    #                 myVO.calculateAll("./VO Practice Sequence/VO Practice Sequence/", totalVid = 701)


    doThese = [(2, 800, 0.05, 50), (2, 1600, 0.05, 50), (3, 800, 0.05, 5), (3, 800, 0.09, 10), (3, 1600, 0.09, 10), (4, 800, 0.01, 15), (4, 800, 0.03, 20), (4, 800, 0.05, 10), (4, 800, 0.07, 10), (4, 800, 0.07, 15), (4, 800, 0.07, 20), (4, 1600, 0.07, 10), (4, 1600, 0.07, 15), (4, 1600, 0.07, 20), (5, 800, 0.03, 50), (5, 800, 0.07, 10), (5, 800, 0.07, 15), (5, 800, 0.09, 50), (5, 1600, 0.03, 5), (5, 1600, 0.03, 50), (5, 1600, 0.07, 10), (5, 1600, 0.07, 15), (5, 1600, 0.09, 50)]

    for skip, corner, quality, minDist in tqdm.tqdm(doThese):
        myVO = VisualOdometry(skip, corner, quality, minDist)

        myVO.calculateAll("./VO Practice Sequence/VO Practice Sequence/", totalVid = 701)
        # myVO.calculateAll("./data/colorVid.avi")
        # myVO.calculateAll("./data/vid_0/", totalVid = 208)