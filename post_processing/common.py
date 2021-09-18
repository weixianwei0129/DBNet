import cv2
import pyclipper
import numpy as np


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(np.min(box[:, 0])).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(np.max(box[:, 0])).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(np.min(box[:, 1])).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(np.max(box[:, 1])).astype(int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def dilate_polygon(box, clipper_ratio=1.5):
    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    distance = area * clipper_ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def norm(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-4) * 255).astype(np.uint8)


def draw_boxes_on_img(img, boxes):
    if len(img.shape) == 4:  # del batch
        img = img[0]
    if img.shape[0] == 3:  # convert CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    boxes = np.array(boxes)
    img = norm(img)
    img = np.ascontiguousarray(img)
    mask = np.zeros_like(img, np.uint8)
    for j in range(boxes.shape[1]):
        box = np.reshape(boxes[0, j], (1, -1, 2)).astype(int)
        cv2.fillPoly(mask, box, (0, 0, 222))
    img = np.clip(mask * .5 + img * .5, 0, 255).astype(np.uint8)
    return img
