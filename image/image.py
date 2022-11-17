import numpy as np
import cv2

# pylint: disable=too-few-public-methods
class Color:
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)


def draw_bboxes(img, bboxes: np.array, color: tuple, width: int = 2):
    """
    Draws bounding boxes on given image

    Parameters
    ----------
    img :
        Image to draw bounding boxes on
    bboxes :
        A two dimensional array of corner points where each row is in format
        [x1, y1, x2, y2]

    Returns
    -------
    Image with drawn bounding boxes
    """

    img_bbox = img.copy()

    if len(bboxes.shape) != 2:
        raise ValueError(f"bboxes must be 2-dimensional, are {len(bboxes.shape)}")

    if bboxes.shape[1] != 4:
        raise ValueError(f"bboxes second dimension must be 4, is {bboxes.shape[1]}")

    for bbox in bboxes:
        cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            width,
        )

    return img_bbox
