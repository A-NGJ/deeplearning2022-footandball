import os
import sys

import cv2
import numpy as np
import pandas as pd

from data.augmentation import PLAYER_LABEL, BALL_LABEL
from data.soccer_net import SoccerNet
from image import image

PACKAGE_PATH = "label"
SOCCER_NET_PATH = os.path.expandvars("${DATA_PATH}/soccer_net/tracking/train/")


def key_to_label(key: str):
    if key == 112:  # key == "p"
        return PLAYER_LABEL
    if key == 98:  # key == "b"
        return BALL_LABEL
    return -1


def load(filename: str):
    with open(filename, "r", encoding="utf-8") as rfile:
        df = pd.read_csv(rfile)

    return df


def save(filename: str, df: pd.DataFrame):
    with open(filename, "w", encoding="utf-8") as wfile:
        df.to_csv(wfile, index=False)


def update_labels(filename: str):
    ...


def run():

    columns = ["image", "bbox", "label"]
    sn = SoccerNet(SOCCER_NET_PATH)
    sn.collect()

    file_dir = ""
    df = None
    filename = ""

    for img_path, annot in zip(sn.image_list, sn.gt):

        new_file_dir = img_path.split("/")[0]
        if new_file_dir != file_dir:
            if filename:
                save(filename, df)

            file_dir = new_file_dir

            filename = os.path.join(PACKAGE_PATH, f"labels_{file_dir}.csv")
            if not os.path.exists(filename):
                with open(filename, "w", encoding="utf-8") as wfile:
                    df = pd.DataFrame(columns=columns)
                    df.to_csv(wfile, index=False)

            df = load(filename)
            images = set(df["image"])

        if img_path in images:
            continue

        img = cv2.imread(SOCCER_NET_PATH + img_path)

        img_data = []
        for i, bbox in enumerate(annot):
            while True:
                cv2.imshow(
                    img_path,
                    image.draw_bboxes(img, np.array([bbox]), image.Color.BLUE, width=4),
                )
                key = cv2.waitKey(0)
                if key == 27:  # esc
                    save(filename, df)
                    sys.exit(0)

                key = key_to_label(key)
                if key > 0:
                    break
                print("invalid key, choose from {b - ball, p - player, esc - exit}")

            img_data.append([img_path, i, key])
            cv2.destroyAllWindows()

        df = pd.concat([df, pd.DataFrame(img_data, columns=columns)])
