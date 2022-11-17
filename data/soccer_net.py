from collections import defaultdict
import csv
import os

# pylint: disable=too-many-locals
def collect(directory: str) -> defaultdict:
    """
    Collects data from soccer net data sets

    Parameters
    ----------
    Data directory INSIDE soccer net directory. E.g. test, train

    Requires
    --------
    ENV SOCCER_NET_PATH : absolute path to soccer net directory set as environment variable

    Returns
    -------
    defaultdict:
        Dictionary with image paths as keys and annotations of corner points as values (x1, y1, x2, y2)
    """

    path = os.path

    soccer_net_path = os.getenv("SOCCER_NET_PATH")
    assert soccer_net_path, "missing env SOCCER_NET_PATH"

    abs_path = path.join(soccer_net_path, directory)

    annotation_files = []
    image_annotations = defaultdict(list)

    for subdir in os.listdir(abs_path):
        if "SNMOT" in subdir:
            for file_ in os.listdir(path.join(abs_path, subdir, "det")):
                annotation_files.append(path.join(abs_path, subdir, "det", file_))

    for annotated_file in annotation_files:
        st, sample = annotated_file.split("/")[-4:-2]
        imgpath = "{0:s}/{1:s}/img1/{2:0>6d}.jpg"
        with open(annotated_file, "r", encoding="utf-8") as rfile:
            for row in csv.reader(rfile.readlines()):
                frame, _, x, y, w, h = [int(x) for x in row[:6]]
                image_annotations[imgpath.format(st, sample, frame)].append(
                    (x, y, x + w, y + h)
                )

    return image_annotations
