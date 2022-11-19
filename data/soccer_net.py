from collections import defaultdict
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data.augmentation import (
    PLAYER_LABEL,
    BALL_LABEL,
)


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
        image = Image.open(img_path)
        boxes, labels = self.get_annotations(idx)
        if self.transform is not None:
            image, boxes, labels = self.transform((image, boxes, labels))

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, boxes, labels

    # pylint: disable=too-many-locals
    def collect(self) -> dict:
        """
        Collects data from soccer net data sets

        Requires
        --------
        ENV SOCCER_NET_PATH : absolute path to soccer net directory set as environment variable
        """

        path = os.path

        soccer_net_path = os.getenv("SOCCER_NET_PATH")
        assert soccer_net_path, "missing env SOCCER_NET_PATH"

        abs_path = path.join(soccer_net_path, self.data_path)

        annotation_files = []
        image_annotations = defaultdict(list)

        for subdir in Path(abs_path).glob("SNMOT*"):
            for file_ in os.listdir(path.join(abs_path, subdir, "det")):
                annotation_files.append(path.join(abs_path, subdir, "det", file_))

        for annotated_file in annotation_files:
            st, sample = annotated_file.split("/")[-4:-2]
            imgpath = "{0:s}/{1:s}/img1/{2:0>6d}.jpg"

            with open(annotated_file, "r", encoding="utf-8") as rfile:
                for row in csv.reader(rfile.readlines()):
                    frame, _, x, y, w, h = [int(x) for x in row[:6]]
                    image_annotations[imgpath.format(st, sample, frame)].append(
                        [x, y, x + w, y + h]
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

        min_ = np.inf
        ball_idx = 0
        # find the ball as a smallest bounding box
        for (x1, y1, x2, y2) in self.gt[idx]:
            size = x2 - x1 + y2 - y1
            if size < min_:
                min_ = size
                ball_idx = idx

        # Add annotations
        for i, (x1, y1, x2, y2) in enumerate(self.gt[idx]):
            bboxes.append((x1, y1, x2, y2))
            if i == ball_idx:
                labels.append(BALL_LABEL)
            else:
                labels.append(PLAYER_LABEL)

        return np.array(bboxes, dtype=float), np.array(labels, dtype=np.int64)
