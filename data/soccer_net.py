from collections import defaultdict
import csv
import os
from pathlib import Path
import typing as t

import numpy as np
from PIL import Image
import torch

from data import augmentation


class SoccerNet(torch.utils.data.Dataset):
    def __init__(self, data_path: str, transform=None):
        self.image_list = []
        self.gt = []
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
        img_path = self.image_list[idx]
        image = Image.open(os.path.join(self.data_path, img_path))
        boxes, labels = self.get_annotations(idx)
        if self.transform is not None:
            image, boxes, labels = self.transform((image, boxes, labels))

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, boxes, labels

    # pylint: disable=too-many-locals
    def collect(self, ids: t.Optional[t.List[str]] = None) -> dict:
        """
        Collects data from soccer net data sets

        Parameters
        ----------
        ids : Optional[List[str]]
            List of dataset ids to be included in the merged dataset
        """

        path = os.path

        if not ids:
            ids = [subdir.name for subdir in Path(self.data_path).glob("SNMOT*")]

        annotation_files = []
        image_annotations = defaultdict(list)

        for subdir in Path(self.data_path).glob("SNMOT*"):
            if subdir.name in ids:
                annotation_files.append(
                    path.join(self.data_path, subdir, "det", "det.txt")
                )

        for annotated_file in annotation_files:
            sample = annotated_file.split(os.sep)[-3]
            imgpath = "{0:s}/img1/{1:0>6d}.jpg"

            with open(annotated_file, "r", encoding="utf-8") as rfile:
                for row in csv.reader(rfile.readlines()):
                    # frame, _, x, y, w, h = [int(x) for x in row[:6]]
                    values = list(map(int, row))
                    # values[0] - frame
                    # values[2] - x
                    # values[3] - y
                    # values[4] - w
                    # values[5] - h
                    # values[-1] - label {ball,player,none}
                    image_annotations[imgpath.format(sample, values[0])].append(
                        [
                            values[2],
                            values[3],
                            values[2] + values[4],
                            values[3] + values[5],
                            values[-1],
                        ]
                    )

        for img, annot in image_annotations.items():
            self.gt.append(np.array(annot))
            self.image_list.append(img)

    def get_annotations(self, idx: int):
        """
        Prepare annotations as list of boxes (xmin, ymin, xmax, ymax) in pixel coordinates
        and torch int64 tensor of corresponding labels
        """

        bboxes = []
        labels = []

        # Add annotations
        for _, (x1, y1, x2, y2, label) in enumerate(self.gt[idx]):
            bboxes.append((x1, y1, x2, y2))
            if label > 0:
                labels.append(label)
            else:
                size = abs(x2 - x1) * abs(y2 - y1)
                # threshold = avg ball size + std deviation
                if size < 1002:
                    labels.append(augmentation.BALL_LABEL)
                else:
                    labels.append(augmentation.PLAYER_LABEL)

        return np.array(bboxes, dtype=float), np.array(labels, dtype=np.int64)


def create_soccer_net_dataset(
    path: str, ids: t.Optional[t.List[str]] = None, mode: str = "train"
):
    """
    Create merged dataset with applied transform.

    Parameters
    ----------
    path : str
        Path to dataset
    mode : str {train|val}
        Dataset mode. Augmentation is applied for train.
    ids : Optional[List[str]]
        List of dataset ids to be included in the merged dataset

    Returns
    -------
    dataset
        Merged dataset
    """
    assert mode in ("train", "val")
    assert os.path.exists(path), f"cannot find dataset {path}"

    image_size = (720, 1280)
    if mode == "train":
        transform = augmentation.TrainAugmentation(image_size)
    else:
        # mode == "val"
        transform = augmentation.NoAugmentation(image_size)

    soccer_net = SoccerNet(path, transform)
    soccer_net.collect(ids)

    return soccer_net
