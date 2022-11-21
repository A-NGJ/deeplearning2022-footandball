# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import os
import configparser
import time


class Params:
    def __init__(self, path):
        assert os.path.exists(path), f"Cannot find configuration file: {path}"
        self.path = path

        config = configparser.ConfigParser()

        config.read(self.path)
        params = config["DEFAULT"]

        # ISSIA
        self.issia_path = os.path.expandvars(params.get("issia_path", ""))
        self.issia_train_cameras, self.issia_val_cameras = [], []
        if self.issia_path:
            temp = params.get("issia_train_cameras", "1, 2, 3, 4")
            self.issia_train_cameras = [int(e) for e in temp.split(",")]
            temp = params.get("issia_val_cameras", "5, 6")
            self.issia_val_cameras = [int(e) for e in temp.split(",")]

        # SPD BMVC17
        self.spd_path = os.path.expandvars(params.get("spd_path", ""))
        self.spd_set = []
        if self.spd_path:
            temp = params.get("spd_set", "1, 2")
            self.spd_set = [int(e) for e in temp.split(",")]

        # SOCCER_NET
        self.soccer_net_path = os.path.expandvars(params.get("soccer_net_path", ""))
        self.soccer_net_set = []
        if self.soccer_net_path:
            temp = params.get("soccer_net_set")
            if temp is not None:
                self.soccer_net_set = temp.split(",")
            else:
                self.soccer_net_set = None

        self.num_workers = params.getint("num_workers", 0)
        self.batch_size = params.getint("batch_size", 4)
        self.epochs = params.getint("epochs", 20)
        self.lr = params.getfloat("lr", 1e-3)

        self.model = params.get("model", "fb1")
        self.model_name = f"model_{self.model}_{get_datetime()}"

        self._check_params()

    def _check_params(self):
        if not any((self.issia_path, self.spd_path, self.soccer_net_path)):
            raise ValueError("at least one dataset must be provided")
        # ISSIA
        if self.issia_path:
            assert os.path.exists(
                self.issia_path
            ), f"Cannot access ISSIA CNR dataset: {self.issia_path}"
        # SPD BMVC17
        if self.spd_path:
            assert os.path.exists(
                self.spd_path
            ), f"Cannot access SoccerPlayerDetection_bmvc17 dataset: {self.spd_path}"
        # SOCCER_NET
        if self.soccer_net_path:
            assert os.path.exists(
                self.soccer_net_path
            ), f"Cannot access Soccer Net dataset: {self.soccer_net_path}"

        for c in self.issia_train_cameras:
            assert (
                1 <= c <= 6
            ), f"ISSIA CNR camera number must be between 1 and 6. Is: {c}"
        for c in self.issia_val_cameras:
            assert (
                1 <= c <= 6
            ), f"ISSIA CNR camera number must be between 1 and 6. Is: {c}"
        for c in self.spd_set:
            assert c == 1 or c == 2, f"SPD dataset number must be 1 or 2. Is: {c}"

    def print(self):
        print("Parameters:")
        param_dict = vars(self)
        for e in param_dict:
            print(f"{e}: {param_dict[e]}")
        print("")


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")
