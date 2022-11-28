# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Run FootAndBall detector on ISSIA-CNR Soccer videos
#

import os
import argparse

import torch
import cv2
import tqdm
import json

from misc import utils
from network import footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL
from data import soccer_net
import numpy as np
import json
import pickle

TEST_DIR = os.path.expandvars("${REPO}/runs/test")

DATAPATH = r"C:\Users\psaff\Desktop\MasterAutonomousSystems\4semester\Deep Learning\project\datasets\SoccerNet\tracking\test\test"
DATA_PATH = r"C:\Users\psaff\Desktop\MasterAutonomousSystems\4semester\Deep Learning\project\datasets"
DATA = r"C:\Users\psaff\Desktop\MasterAutonomousSystems\4semester\Deep Learning\project\datasets\SoccerNet\tracking\test\test\SNMOT-116\img1"

soccerNet = soccer_net.SoccerNet(DATAPATH)
soccerNet.collect(["SNMOT-116"])
print(soccerNet.gt[0])
print("lenght: ", len(soccerNet.gt[0]))
# gt = soccerNet.get_annotations(1)
# print(gt)


def getDet(data):
    # print('Detections: ', data )
    IMG_HEIGHT = 1920
    IMG_WIDTH = 1080
    gt = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=bool)
    cntgt = 0

    for i in range(0, len(data) - 1):
        X1 = int(data[i][0])
        Y1 = int(data[i][1])
        X2 = int(data[i][2])
        Y2 = int(data[i][3])
        if X1 > IMG_HEIGHT:
            X1 = IMG_HEIGHT
        if X2 > IMG_HEIGHT:
            X2 = IMG_HEIGHT
        if Y1 > 1080:
            Y1 = 1080
        if Y2 > 1080:
            Y2 = 1080  # for SMOT-162 test set, img=454 this condition makes sense
        # print(X1, ' ', Y1, ' ', X2, ' ', Y2 )
        cntgt += 1
        # print('X: ', X, 'Y: ', Y, 'width: ',width, 'height: ', height)
        for i in range(X1, X2 - 1):
            for j in range(Y1, Y2 - 1):
                # print(i, j)
                gt[i][j] = 1
    # gt - ground truth array, cntgt - number of objects
    # print(type(gt))
    # print(np.size(gt))
    # print(gt.ndim)
    # print(gt.shape)
    return gt, cntgt


def getGT(data):
    # print('soccernet GT: ', data )
    IMG_HEIGHT = 1920
    IMG_WIDTH = 1080
    gt = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=bool)
    cntgt = 0

    for i in range(0, len(data) - 1):
        X1 = int(data[i][0])
        Y1 = int(data[i][1])
        X2 = int(data[i][2])
        Y2 = int(data[i][3])
        # print(X1, ' ', Y1, ' ', X2, ' ', Y2 )
        cntgt += 1
        # print('X: ', X, 'Y: ', Y, 'width: ',width, 'height: ', height)
        for i in range(X1, X2):
            for j in range(Y1, Y2):
                # print(i, j)
                gt[i][j] = 1
    # gt - ground truth array, cntgt - number of objects
    # print(type(gt))
    # print(np.size(gt))
    # print(gt.ndim)
    # print(gt.shape)
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


def draw_bboxes(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(
        detections["boxes"], detections["labels"], detections["scores"]
    ):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                image,
                "{:0.2f}".format(score),
                (int(x1), max(0, int(y1) - 10)),
                font,
                1,
                color,
                2,
            )

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(
                image,
                "{:0.2f}".format(score),
                (max(0, int(x - radius)), max(0, (y - radius - 10))),
                font,
                1,
                color,
                2,
            )

    return image


def run_detector_soccerNet(model, args):
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    if args.device == "cpu":
        print("Loading CPU weights...")
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print("Loading GPU weights...")
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    cap = cv2.VideoCapture(
        r"C:\Users\psaff\Desktop\MasterAutonomousSystems\4semester\Deep Learning\project\datasets\SoccerNet\tracking\test\test\SNMOT-116\img1\%06d.jpg",
        cv2.CAP_IMAGES,
    )
    fps = 30  # cap.get(cv2.CAP_PROP_FPS)

    (frame_width, frame_height) = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print("width: ", frame_width, "height: ", frame_height)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sequence = cv2.VideoWriter(
        args.out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    print(f"Processing video: {args.path}")
    pbar = tqdm.tqdm(total=n_frames)

    counter = -1
    print("FRAMES to process: ", n_frames)
    iou = np.zeros(n_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # End of video
            break
        counter += 1
        # Convert color space from BGR to RGB, convert to tensor and normalize
        img_tensor = augmentations.numpy2tensor(frame)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
            detections = model(img_tensor)[0]
            gt, cntgt = getGT(soccerNet.gt[counter])
            # detections = getGT(detections)

            det = detections["boxes"]
            det, cntdet = getDet(detections["boxes"])
            # print("DETECTIONS: ", det)
            # print('GT: ', gt)

            iou[counter - 1] = IoU(gt, det)
            # print('IOU: ', iou)

        frame = draw_bboxes(frame, detections)
        out_sequence.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out_sequence.release()
    metric = iou
    return metric


def run_detector(model, args):
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    if args.device == "cpu":
        print("Loading CPU weights...")
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print("Loading GPU weights...")
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    sequence = cv2.VideoCapture(args.path)
    fps = sequence.get(cv2.CAP_PROP_FPS)
    (frame_width, frame_height) = (
        int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sequence = cv2.VideoWriter(
        args.out_video,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (frame_width, frame_height),
    )

    print(f"Processing video: {args.path}")
    pbar = tqdm.tqdm(total=n_frames)
    while sequence.isOpened():
        ret, frame = sequence.read()
        if not ret:
            # End of video
            break

        # Convert color space from BGR to RGB, convert to tensor and normalize
        img_tensor = augmentations.numpy2tensor(frame)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
            detections = model(img_tensor)[0]

        frame = draw_bboxes(frame, detections)
        out_sequence.write(frame)
        pbar.update(1)

    pbar.close()
    sequence.release()
    out_sequence.release()


if __name__ == "__main__":
    print("Run FootAndBall detector on input video")

    # if not "DATA_PATH" in os.environ:
    #    raise EnvironmentError("missing DATA_PATH environmental variable")

    # if not "REPO" in os.environ:
    #    raise EnvironmentError("missing REPO environmental variable")

    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to video", type=str, required=True)
    parser.add_argument("--model", help="model name", type=str, default="fb1")
    parser.add_argument(
        "--weights", help="path to model weights", type=str, required=True
    )
    parser.add_argument(
        "--ball_threshold",
        help="ball confidence detection threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--player_threshold",
        help="player confidence detection threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--out_video",
        help="path to video with detection results",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--device", help="device (CPU or CUDA)", type=str, default="cuda:0"
    )
    parser.add_argument(
        "--run-dir",
        help="[Optional] Directory for saving test data; default: YYMMDD_HHMM",
        required=False,
        default=utils.get_current_time(),
    )
    args = parser.parse_args()

    print(f"Video path: {args.path}")
    print(f"Model: {args.model}")
    print(f"Model weights path: {args.weights}")
    print(f"Ball confidence detection threshold [0..1]: {args.ball_threshold}")
    print(f"Player confidence detection threshold [0..1]: {args.player_threshold}")
    print(f"Output video path: {args.out_video}")
    print(f"Device: {args.device}")
    print("")

    assert os.path.exists(
        args.weights
    ), f"Cannot find FootAndBall model weights: {args.weights}"
    assert os.path.exists(args.path), f"Cannot open video: {args.path}"

    model = footandball.model_factory(
        args.model,
        "detect",
        ball_threshold=args.ball_threshold,
        player_threshold=args.player_threshold,
    )

    # general run history directory
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    # run specific history directory
    run_dir = f"{TEST_DIR}/{args.run_dir}"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    args.out_video = os.path.join(run_dir, args.out_video)
    metric = run_detector_soccerNet(model, args)

    pickle.dump(metric, open("metricOLO.p", "wb"))

    with open("metricOLO.json", "w") as outfile:
        json.dump(metric, outfile)

    print("metrics saved!")
