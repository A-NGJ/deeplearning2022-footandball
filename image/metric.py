import typing as t
import numpy as np


def getGT(data, img_height=1920, img_width=1080) -> t.Tuple[np.array, int]:
    """
    Returns
    -------
    gt :
        ground truth array
    cntgt :
        number of objects
    """
    gt = np.zeros((img_height, img_width), dtype=bool)
    cntgt = 0

    for i in range(0, len(data) - 1):
        X1 = min(int(data[i][0]), img_height)
        Y1 = min(int(data[i][1]), img_width)
        X2 = min(int(data[i][2]), img_height)
        Y2 = min(int(data[i][3]), img_width)

        cntgt += 1

        for i in range(X1, X2 - 1):
            for j in range(Y1, Y2 - 1):
                gt[i][j] = 1

    return gt, cntgt


def IoU(result, gt):
    # result, gt - np boolean arrays
    overlap = result * gt  # Logical AND
    union = result + gt  # Logical OR

    IOU = overlap.sum() / float(union.sum())  # Treats "True" as 1,
    # sums number of Trues
    # in overlap and union
    # and divides
    return IOU
