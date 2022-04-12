import tqdm

from VisualOdometry import VisualOdometry

if __name__ == "__main__":
    skipFrames = [2,3,4,5]  # 7#5#3#5 #3
    maxCorners = [800,1600]
    qualitlyLevel = [0.01, 0.03, 0.05, 0.07, 0.09]
    minDistance = [5, 10, 15, 20, 50]

    for skip in tqdm.tqdm(skipFrames):
        for corner in maxCorners:
            for quality in qualitlyLevel:
                for minDist in minDistance:
                    myVO = VisualOdometry(skip, corner, quality, minDist)

                    myVO.calculateAll("./VO Practice Sequence/VO Practice Sequence/")
